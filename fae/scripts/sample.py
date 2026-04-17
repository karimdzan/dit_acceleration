from pathlib import Path

import torch
from PIL import Image

from fae.config import load_yaml
from fae.generators.common import ConditioningBundle, LatentTensorSpec
from fae.scripts.common import (
    build_backbone_from_config,
    build_bridge_from_config,
    build_device,
    build_fae_from_config,
    build_generator_from_config,
    build_pixel_decoder_from_config,
    get_fae_latent_spec,
    get_train_dtype,
    parse_args,
)
from fae.utils.checkpoint import load_checkpoint


def save_image_grid(images: torch.Tensor, path: str | Path) -> None:
    images = images.detach().cpu().clamp(-1.0, 1.0)
    images = ((images + 1) * 127.5).to(torch.uint8)
    b, c, h, w = images.shape
    cols = min(4, b)
    rows = (b + cols - 1) // cols
    canvas = Image.new('RGB', (cols * w, rows * h))
    for idx, image in enumerate(images):
        img = Image.fromarray(image.permute(1, 2, 0).numpy())
        x = (idx % cols) * w
        y = (idx // cols) * h
        canvas.paste(img, (x, y))
    canvas.save(path)


def main():
    args = parse_args('Sample from a generator trained on FAE latents')
    config = load_yaml(args.config)
    device = build_device(config)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    backbone = build_backbone_from_config(config).to(device)
    fae = build_fae_from_config(config, input_dim=backbone.output_dim).to(device)
    fae_state = load_checkpoint(config['stage3']['fae_checkpoint'], map_location='cpu')
    fae.load_state_dict(fae_state['model'])
    fae.eval()

    pixel_decoder = build_pixel_decoder_from_config(config, input_dim=backbone.output_dim).to(device)
    pixel_ckpt = load_checkpoint(config['stage3']['pixel_decoder_checkpoint'], map_location='cpu')
    pixel_decoder.load_state_dict(pixel_ckpt['model'])
    pixel_decoder.eval()

    fae_spec = get_fae_latent_spec(config)
    bridge_enabled = bool(config.get('bridge', {}).get('enabled', True))
    generator_spec = fae_spec if not bridge_enabled else LatentTensorSpec(
        channels=config['generator'].get('in_channels', fae_spec.channels),
        height=config['generator'].get('sample_size', fae_spec.height),
        width=config['generator'].get('sample_size', fae_spec.width),
    )
    generator = build_generator_from_config(config, model_spec=generator_spec).to(device)
    bridge = build_bridge_from_config(config, fae_spec=fae_spec, model_spec=generator.latent_spec()).to(device)

    state = load_checkpoint(config['stage3']['generator_checkpoint'], map_location='cpu')
    generator.load_state_dict(state['model'], strict=False)
    if 'bridge' in state:
        bridge.load_state_dict(state['bridge'])
    generator.eval()
    bridge.eval()
    train_dtype = get_train_dtype(config)

    if train_dtype in (torch.float16, torch.bfloat16):
        fae = fae.to(train_dtype)
        pixel_decoder = pixel_decoder.to(train_dtype)
        generator = generator.to(train_dtype)
        bridge = bridge.to(train_dtype)

    conditioning = None
    if args.prompt and getattr(generator, 'uses_native_prompt_encoder', False):
        conditioning = generator.encode_prompts([args.prompt] * args.num_samples, device=device)
    if args.class_label is not None:
        label_tensor = torch.full((args.num_samples,), int(args.class_label), device=device, dtype=torch.long)
        if conditioning is None:
            conditioning = ConditioningBundle(class_labels=label_tensor)
        else:
            conditioning.class_labels = label_tensor
    elif config['generator']['name'] == 'diffusers_dit':
        raise ValueError('--class-label is required for class-conditional DiT sampling.')

    with torch.no_grad():
        latents = generator.sample_latents(
            args.num_samples,
            device=device,
            conditioning=conditioning,
            num_steps=config.get('sampling', {}).get('num_steps', 30),
        )
        z_tokens = bridge.to_fae_tokens(latents)
        features = fae.decode(z_tokens)
        images = pixel_decoder(features)

    out_dir = Path(config.get('sampling', {}).get('output_dir', 'samples'))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'sample_grid.png'
    save_image_grid(images, out_path)
    print(out_path)


if __name__ == '__main__':
    main()
