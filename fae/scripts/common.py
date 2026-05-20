import contextlib
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from fae.backbones import build_backbone
from fae.data import ImageFolderWithOptionalCaptions, collate_samples
from fae.generators import build_generator_backend
from fae.generators.common import LatentTensorSpec
from fae.models import (
    ClassConditioner,
    RepresentationAutoEncoder,
    FrozenTextConditioner,
    IdentityLatentBridge,
    LatentBridge,
    LatentBridgeSpec,
    ViTPixelDecoder,
)
from fae.utils.checkpoint import extract_model_state_dict, load_checkpoint


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
    return torch.device(requested)


def _get_autoencoder_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("rae") or config.get("fae") or {})


def _load_latent_stats(stats_path: str | None, input_dim: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not stats_path:
        return None, None
    state = torch.load(Path(stats_path), map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"Latent stats file must contain a dict, got {type(state)}.")

    mean = state.get("mean", state.get("feature_mean", state.get("latent_mean")))
    var = state.get("var", state.get("variance", state.get("latent_var")))
    std = state.get("std", state.get("feature_std"))
    if mean is None:
        raise KeyError(f"Latent stats file {stats_path} must contain a mean tensor.")
    if var is None:
        if std is None:
            raise KeyError(f"Latent stats file {stats_path} must contain var or std.")
        std_t = torch.as_tensor(std, dtype=torch.float32)
        var = std_t.square()

    mean_t = torch.as_tensor(mean, dtype=torch.float32)
    var_t = torch.as_tensor(var, dtype=torch.float32)

    if mean_t.ndim == 1:
        if mean_t.numel() != input_dim or var_t.numel() != input_dim:
            raise ValueError(
                f"Channel latent stats mismatch for {stats_path}: expected {input_dim} dims, "
                f"got mean={mean_t.numel()} var={var_t.numel()}."
            )
        return mean_t, var_t

    if mean_t.ndim == 4 and mean_t.shape[0] == 1:
        mean_t = mean_t.squeeze(0)
        var_t = var_t.squeeze(0)
    if mean_t.ndim != 3 or mean_t.shape[0] != input_dim:
        raise ValueError(
            f"Spatial latent stats mismatch for {stats_path}: expected [C,H,W] with C={input_dim}, got {tuple(mean_t.shape)}."
        )
    if tuple(mean_t.shape) != tuple(var_t.shape):
        raise ValueError(f"Latent stats mismatch for {stats_path}: {tuple(mean_t.shape)} vs {tuple(var_t.shape)}")
    return mean_t, var_t


def build_fae_from_config(config: dict[str, Any], input_dim: int) -> RepresentationAutoEncoder:
    ae_cfg = _get_autoencoder_cfg(config)
    pixel_cfg = config.get("pixel_decoder", {})
    latent_mean, latent_var = _load_latent_stats(
        ae_cfg.get("latent_stats_path") or ae_cfg.get("feature_stats_path"),
        input_dim,
    )
    decoder_hidden_dim = pixel_cfg.get("hidden_dim", ae_cfg.get("decoder_hidden_dim", input_dim))
    decoder_layers = pixel_cfg.get("num_layers", ae_cfg.get("decoder_layers", 24))
    decoder_heads = pixel_cfg.get("num_heads", ae_cfg.get("decoder_heads", 16))
    decoder_head_dim = pixel_cfg.get("head_dim", ae_cfg.get("decoder_head_dim"))
    decoder_mlp_ratio = pixel_cfg.get("mlp_ratio", ae_cfg.get("decoder_mlp_ratio", 4.0))
    decoder_use_rope_2d = pixel_cfg.get("use_rope_2d", ae_cfg.get("decoder_use_rope_2d", True))

    return RepresentationAutoEncoder(
        input_dim=input_dim,
        image_size=pixel_cfg.get("image_size", 256),
        patch_size=pixel_cfg.get("patch_size", 16),
        decoder_hidden_dim=decoder_hidden_dim,
        decoder_layers=decoder_layers,
        decoder_heads=decoder_heads,
        decoder_head_dim=decoder_head_dim,
        decoder_mlp_ratio=decoder_mlp_ratio,
        decoder_use_rope_2d=decoder_use_rope_2d,
        rope_base=ae_cfg.get("rope_base", 10000.0),
        noise_tau=ae_cfg.get("noise_tau", 0.0),
        reshape_to_2d=ae_cfg.get("reshape_to_2d", True),
        normalize_latents=ae_cfg.get("normalize_latents", ae_cfg.get("normalize_features", False)),
        latent_mean=latent_mean,
        latent_var=latent_var,
        latent_eps=ae_cfg.get("latent_eps", ae_cfg.get("feature_eps", 1e-6)),
    )


def build_pixel_decoder_from_config(config: dict[str, Any], input_dim: int) -> ViTPixelDecoder:
    return build_fae_from_config(config, input_dim=input_dim).decoder


def _infer_fae_hw(config: dict[str, Any], backbone=None) -> tuple[int, int]:
    ae_cfg = _get_autoencoder_cfg(config)
    if ae_cfg.get('latent_height') and ae_cfg.get('latent_width'):
        return int(ae_cfg['latent_height']), int(ae_cfg['latent_width'])
    if ae_cfg.get('latent_grid_size'):
        side = int(ae_cfg['latent_grid_size'])
        return side, side
    if backbone is not None and getattr(backbone, 'latent_hw', None) is not None:
        return tuple(int(v) for v in backbone.latent_hw)
    stats_path = ae_cfg.get('latent_stats_path') or ae_cfg.get('feature_stats_path')
    if stats_path:
        state = torch.load(Path(stats_path), map_location='cpu')
        mean = state.get('mean', state.get('feature_mean', state.get('latent_mean')))
        if isinstance(mean, torch.Tensor) and mean.ndim in {3, 4}:
            if mean.ndim == 4:
                mean = mean.squeeze(0)
            return int(mean.shape[-2]), int(mean.shape[-1])
    image_size = int(config.get('pixel_decoder', {}).get('image_size', 256))
    patch_size = int(config.get('pixel_decoder', {}).get('patch_size', 16))
    side = image_size // patch_size
    return side, side


def get_fae_latent_spec(config: dict[str, Any], input_dim: int | None = None, backbone=None) -> LatentTensorSpec:
    fae_h, fae_w = _infer_fae_hw(config, backbone=backbone)
    encoder_cfg = config.get("encoder", {})
    ae_cfg = _get_autoencoder_cfg(config)
    channels = input_dim or encoder_cfg.get("output_dim") or ae_cfg.get("latent_dim")
    if channels is None:
        raise ValueError("Could not infer latent channel count from config. Pass input_dim when constructing the model.")
    return LatentTensorSpec(channels=int(channels), height=fae_h, width=fae_w)


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
    return DataLoader(
        dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=train_cfg.get('shuffle', True),
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
    model.load_state_dict(extract_model_state_dict(state, prefer_ema=True))
    if optimizer is not None and 'optimizer' in state:
        optimizer.load_state_dict(state['optimizer'])
    if scaler is not None and state.get('scaler') is not None:
        scaler.load_state_dict(state['scaler'])
    if extra_state_loaders is not None:
        for key, fn in extra_state_loaders.items():
            if key in state and state[key] is not None:
                fn(state[key])
    return int(state.get('epoch', 0)), state
