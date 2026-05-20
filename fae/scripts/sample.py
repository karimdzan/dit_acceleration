from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from fae.generators.common import ConditioningBundle, LatentTensorSpec
from fae.scripts.common import (
    build_backbone_from_config,
    build_bridge_from_config,
    build_device,
    build_fae_from_config,
    build_generator_from_config,
    get_fae_latent_spec,
    get_train_dtype,
)
from fae.utils.checkpoint import extract_model_state_dict, load_checkpoint
from fae.utils.pretrained import initialize_rae_from_pretrained, resolve_rae_assets


def save_image_grid(images: torch.Tensor, path: str | Path) :
    images = images.detach().cpu().float()
    if float(images.amin()) < -0.05 or float(images.amax()) > 1.05:
        images = images.clamp(-1.0, 1.0).add(1.0).mul(0.5)
    else:
        images = images.clamp(0.0, 1.0)
    images = (images * 255.0).round().to(torch.uint8)
    b, c, h, w = images.shape
    cols = min(4, b)
    rows = (b + cols - 1) // cols
    canvas = Image.new('RGB', (cols * w, rows * h))
    for idx, image in enumerate(images):
        img = Image.fromarray(image.permute(1, 2, 0).numpy())
        canvas.paste(img, ((idx % cols) * w, (idx // cols) * h))
    canvas.save(path)


@hydra.main(version_base=None, config_path=None)
def main(cfg: DictConfig) :
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    sample_cfg = config.get('sample', {})
    device = build_device(config)

    seed = sample_cfg.get('seed', 1234)
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    backbone = build_backbone_from_config(config).to(device)
    autoencoder = build_fae_from_config(config, input_dim=backbone.output_dim).to(device)
    pretrained_info = initialize_rae_from_pretrained(autoencoder, config)
    if pretrained_info.get("loaded"):
        print(
            f"Initialized RAE from pretrained {pretrained_info['mode']} weights: "
            f"{pretrained_info.get('autoencoder_checkpoint') or pretrained_info.get('decoder_path')}"
        )
    autoencoder.eval()

    ae_spec = get_fae_latent_spec(config, input_dim=backbone.output_dim, backbone=backbone)
    bridge_enabled = bool(config.get('bridge', {}).get('enabled', True))
    generator_spec = ae_spec if not bridge_enabled else LatentTensorSpec(
        channels=config['generator'].get('in_channels', ae_spec.channels),
        height=config['generator'].get('sample_size', ae_spec.height),
        width=config['generator'].get('sample_size', ae_spec.width),
    )
    generator = build_generator_from_config(config, model_spec=generator_spec).to(device)
    bridge = build_bridge_from_config(config, fae_spec=ae_spec, model_spec=generator.latent_spec()).to(device)

    assets = resolve_rae_assets(config)
    requested_generator_ckpt = config.get('stage3', {}).get('generator_checkpoint')
    generator_ckpt = None
    if requested_generator_ckpt and Path(requested_generator_ckpt).exists():
        generator_ckpt = requested_generator_ckpt
    elif assets.generator_path:
        generator_ckpt = assets.generator_path
    elif requested_generator_ckpt:
        generator_ckpt = requested_generator_ckpt
    if not generator_ckpt:
        raise ValueError('generator checkpoint is required for sampling.')
    state = load_checkpoint(generator_ckpt, map_location='cpu')
    generator.load_state_dict(extract_model_state_dict(state, prefer_ema=True), strict=False)
    if isinstance(state, dict) and 'bridge' in state:
        bridge.load_state_dict(state['bridge'])
    generator.eval()
    bridge.eval()

    train_dtype = get_train_dtype(config)
    if train_dtype in (torch.float16, torch.bfloat16):
        autoencoder = autoencoder.to(train_dtype)
        generator = generator.to(train_dtype)
        bridge = bridge.to(train_dtype)

    num_samples = int(sample_cfg.get('num_samples', 4))
    prompt = sample_cfg.get('prompt')
    class_label = sample_cfg.get('class_label')

    conditioning = None
    if prompt and getattr(generator, 'uses_native_prompt_encoder', False):
        conditioning = generator.encode_prompts([prompt] * num_samples, device=device)
    if class_label is not None:
        label_tensor = torch.full((num_samples,), int(class_label), device=device, dtype=torch.long)
        if conditioning is None:
            conditioning = ConditioningBundle(class_labels=label_tensor)
        else:
            conditioning.class_labels = label_tensor
    elif config['generator']['name'] in {'diffusers_dit', 'internal_dit_dh'}:
        raise ValueError('sample.class_label is required for class-conditional sampling.')

    with torch.no_grad():
        latents = generator.sample_latents(
            num_samples,
            device=device,
            conditioning=conditioning,
            num_steps=config.get('sampling', {}).get('num_steps', 30),
            guidance_scale=config.get('guidance', {}).get('scale', 1.0),
        )
        z_tokens = bridge.to_fae_tokens(latents)
        images = autoencoder.decode(z_tokens)

    out_dir = Path(config.get('sampling', {}).get('output_dir', 'samples'))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'sample_grid.png'
    save_image_grid(images, out_path)
    print(out_path)


if __name__ == '__main__':
    main()
