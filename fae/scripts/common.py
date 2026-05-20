import contextlib
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from fae.backbones import build_backbone
from fae.config import load_yaml
from fae.data import ImageFolderWithOptionalCaptions, collate_samples
from fae.generators import build_generator_backend
from fae.generators.common import LatentTensorSpec
from fae.models import (
    ClassConditioner,
    FeatureAutoEncoder,
    FrozenTextConditioner,
    IdentityLatentBridge,
    LatentBridge,
    LatentBridgeSpec,
    NLayerDiscriminator,
    ViTPixelDecoder,
)
from fae.utils.checkpoint import load_checkpoint
from fae.utils.distributed import get_local_rank, get_rank, get_world_size, is_distributed


def get_train_dtype(config):
    name = str(config.get("train", {}).get("dtype", "fp32")).lower()
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported train.dtype={name}. Use fp32, fp16, or bf16.")
    return mapping[name]


def get_autocast_context(config, device):
    dtype = get_train_dtype(config)
    use_amp = str(device).startswith("cuda") and dtype in (torch.float16, torch.bfloat16)
    if use_amp:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def build_device(config: dict[str, Any]) -> torch.device:
    requested = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    if is_distributed() and str(requested).startswith('cuda'):
        return torch.device('cuda', get_local_rank())
    return torch.device(requested)


def _load_feature_stats(stats_path: str | None, input_dim: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not stats_path:
        return None, None
    state = torch.load(Path(stats_path), map_location='cpu')
    if not isinstance(state, dict):
        raise ValueError(f"Feature stats file must contain a dict, got {type(state)}.")
    mean = state.get('mean', state.get('feature_mean'))
    std = state.get('std', state.get('feature_std'))
    if mean is None or std is None:
        raise KeyError(f"Feature stats file {stats_path} must contain 'mean'/'std' or 'feature_mean'/'feature_std'.")
    mean = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
    std = torch.as_tensor(std, dtype=torch.float32).reshape(-1)
    if mean.numel() != input_dim or std.numel() != input_dim:
        raise ValueError(
            f"Feature stats mismatch for {stats_path}: expected {input_dim} dims, "
            f"got mean={mean.numel()} std={std.numel()}."
        )
    return mean, std


def build_fae_from_config(config: dict[str, Any], input_dim: int) -> FeatureAutoEncoder:
    cfg = config['fae']
    feature_mean, feature_std = _load_feature_stats(cfg.get('feature_stats_path'), input_dim)

    return FeatureAutoEncoder(
        input_dim=input_dim,
        latent_dim=cfg.get('latent_dim', 32),
        encoder_heads=cfg.get('encoder_heads', 8),
        encoder_hidden_dim=cfg.get('encoder_hidden_dim', input_dim),
        encoder_head_dim=cfg.get('encoder_head_dim'),
        encoder_use_rope_2d=cfg.get('encoder_use_rope_2d', False),
        decoder_hidden_dim=cfg.get('decoder_hidden_dim', input_dim),
        decoder_layers=cfg.get('decoder_layers', 6),
        decoder_heads=cfg.get('decoder_heads', 8),
        decoder_head_dim=cfg.get('decoder_head_dim'),
        decoder_use_rope_2d=cfg.get('decoder_use_rope_2d', True),
        rope_base=cfg.get('rope_base', 10000.0),
        kl_weight=cfg.get('kl_weight', 1e-6),
        normalize_features=cfg.get('normalize_features', False),
        feature_mean=feature_mean,
        feature_std=feature_std,
        feature_eps=cfg.get('feature_eps', 1e-6),
    )


def build_pixel_decoder_from_config(config: dict[str, Any], input_dim: int) -> ViTPixelDecoder:
    cfg = config['pixel_decoder']
    decoder = ViTPixelDecoder(
        input_dim=input_dim,
        image_size=cfg.get('image_size', 256),
        patch_size=cfg.get('patch_size', 16),
        hidden_dim=cfg.get('hidden_dim', 1024),
        num_layers=cfg.get('num_layers', 12),
        num_heads=cfg.get('num_heads', 16),
        head_dim=cfg.get('head_dim'),
        mlp_ratio=cfg.get('mlp_ratio', 4.0),
        use_rope_2d=cfg.get('use_rope_2d', True),
        rope_base=cfg.get('rope_base', config.get('fae', {}).get('rope_base', 10000.0)),
    )
    if cfg.get("ckpt", None):
        ckpt = torch.load(cfg["ckpt"], map_location='cpu')
        decoder.load_state_dict(ckpt["model"])
    return decoder


def _infer_fae_hw(config: dict[str, Any]) -> tuple[int, int]:
    image_size = int(config.get('pixel_decoder', {}).get('image_size', 256))
    patch_size = int(config.get('encoder', {}).get('patch_size', 16) or 16)
    side = image_size // patch_size
    return side, side


def get_fae_latent_spec(config: dict[str, Any]) -> LatentTensorSpec:
    fae_h, fae_w = _infer_fae_hw(config)
    return LatentTensorSpec(
        channels=config['fae'].get('latent_dim', 32),
        height=fae_h,
        width=fae_w,
    )


def build_bridge_from_config(config: dict[str, Any], fae_spec: LatentTensorSpec, model_spec: LatentTensorSpec):
    bridge_cfg = config.get('bridge', {})
    spec = LatentBridgeSpec(
        fae_dim=fae_spec.channels,
        fae_height=fae_spec.height,
        fae_width=fae_spec.width,
        model_channels=model_spec.channels,
        model_height=model_spec.height,
        model_width=model_spec.width,
    )
    bridge_enabled = bool(bridge_cfg.get('enabled', True))
    same_shape = (
        spec.fae_dim == spec.model_channels
        and spec.fae_height == spec.model_height
        and spec.fae_width == spec.model_width
    )
    if (not bridge_enabled) or same_shape:
        return IdentityLatentBridge(spec)
    return LatentBridge(
        spec,
        hidden_channels=bridge_cfg.get('hidden_channels'),
        num_res_blocks=bridge_cfg.get('num_res_blocks', 2),
        resize_mode=bridge_cfg.get('resize_mode', 'bilinear'),
    )


def build_generator_from_config(config: dict[str, Any], model_spec: LatentTensorSpec | None = None):
    return build_generator_backend(config, bridge_spec=model_spec)


def maybe_build_conditioners(config: dict[str, Any], device: torch.device):
    class_conditioner = None
    text_conditioner = None
    cond_cfg = config.get('conditioning', {})
    if cond_cfg.get('type') in {'class', 'both'} and config['generator']['name'] == 'internal_dit':
        class_conditioner = ClassConditioner(cond_cfg['num_classes'], config['generator'].get('cond_dim', 1024)).to(device)
    if cond_cfg.get('type') in {'text', 'both'} and config['generator']['name'] == 'internal_dit':
        text_conditioner = FrozenTextConditioner(
            model_name=cond_cfg.get('text_model_name', 'google-t5/t5-base'),
            out_dim=config['generator'].get('cond_dim', 1024),
            max_length=cond_cfg.get('max_length', 64),
        ).to(device)
    return class_conditioner, text_conditioner


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_dataloader(config: dict[str, Any]) -> DataLoader:
    data_cfg = config['data']
    train_cfg = config['train']
    dataset = ImageFolderWithOptionalCaptions(
        root=data_cfg['root'],
        captions_jsonl=data_cfg.get('captions_jsonl'),
        max_samples=data_cfg.get('max_samples'),
        return_none_on_error=data_cfg.get('return_none_on_error', True),
        warn_limit=data_cfg.get('warn_limit', 50),
        load_truncated_images=data_cfg.get('load_truncated_images', True),
    )
    num_workers = int(train_cfg.get('num_workers', 4))
    shuffle = bool(train_cfg.get('shuffle', True))
    sampler = None
    if is_distributed():
        sampler = DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=shuffle,
            drop_last=train_cfg.get('drop_last', False),
        )
    return DataLoader(
        dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=train_cfg.get('pin_memory', True),
        persistent_workers=(num_workers > 0 and train_cfg.get('persistent_workers', True)),
        prefetch_factor=train_cfg.get('prefetch_factor', 2) if num_workers > 0 else None,
        collate_fn=collate_samples,
        drop_last=train_cfg.get('drop_last', False),
    )


def build_backbone_from_config(config: dict[str, Any]):
    enc = dict(config['encoder'])
    name = enc.pop('name')
    return build_backbone(name, **enc)


def get_grad_scaler(config):
    dtype = get_train_dtype(config)
    return torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))


def maybe_get_latest_checkpoint(output_dir: str | Path) -> Path | None:
    path = Path(output_dir) / 'latest.pt'
    return path if path.exists() else None


def maybe_load_resume(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    resume_path: str | None,
    scaler: torch.amp.GradScaler | None = None,
    extra_state_loaders: dict[str, callable] | None = None,
):
    if resume_path is None:
        return 0, None
    state = load_checkpoint(resume_path, map_location='cpu')
    model.load_state_dict(state['model'])
    if optimizer is not None and 'optimizer' in state:
        optimizer.load_state_dict(state['optimizer'])
    if scaler is not None and state.get('scaler') is not None:
        scaler.load_state_dict(state['scaler'])
    if extra_state_loaders is not None:
        for key, fn in extra_state_loaders.items():
            if key in state and state[key] is not None:
                fn(state[key])
    return int(state.get('epoch', 0)), state
