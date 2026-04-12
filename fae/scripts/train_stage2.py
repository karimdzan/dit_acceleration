from pathlib import Path

import torch

from fae.config import load_yaml
from fae.models import NLayerDiscriminator
from fae.scripts.common import (
    build_backbone_from_config,
    build_dataloader,
    build_device,
    build_fae_from_config,
    build_pixel_decoder_from_config,
    get_autocast_context,
    get_grad_scaler,
    get_train_dtype,
    maybe_get_latest_checkpoint,
    maybe_load_resume,
    parse_args,
)
from fae.training import train_stage2_epoch
from fae.utils.checkpoint import load_checkpoint, save_checkpoint
from fae.utils.losses import VGGPerceptualLoss


def main():
    args = parse_args('Train pixel decoder (stage 2)')
    config = load_yaml(args.config)
    device = build_device(config)

    dataloader = build_dataloader(config)
    backbone = build_backbone_from_config(config).to(device)

    fae = build_fae_from_config(config, input_dim=backbone.output_dim).to(device)
    fae_state = load_checkpoint(config['stage2']['fae_checkpoint'], map_location='cpu')
    fae.load_state_dict(fae_state['model'])
    fae.eval()

    pixel_decoder = build_pixel_decoder_from_config(config, input_dim=backbone.output_dim).to(device)
    optimizer = torch.optim.AdamW(
        pixel_decoder.parameters(),
        lr=config['train'].get('lr', 1e-4),
        betas=(0.9, 0.999),
        weight_decay=config['train'].get('weight_decay', 0.05),
    )

    discriminator = None
    disc_optimizer = None
    if config['stage2'].get('use_gan', False):
        discriminator = NLayerDiscriminator().to(device)
        disc_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=config['train'].get('disc_lr', 1e-4))
    perceptual_loss = VGGPerceptualLoss().to(device)

    scaler = get_grad_scaler(config)
    autocast_context = get_autocast_context(config, device)
    train_dtype = get_train_dtype(config)
    if train_dtype in (torch.float16, torch.bfloat16):
        fae = fae.to(train_dtype)
        pixel_decoder = pixel_decoder.to(train_dtype)

    output_dir = Path(config['train'].get('output_dir', 'checkpoints/stage2'))
    resume_path = args.resume or str(maybe_get_latest_checkpoint(output_dir)) if maybe_get_latest_checkpoint(output_dir) else None
    extra_loaders = {}
    if discriminator is not None:
        extra_loaders['discriminator'] = discriminator.load_state_dict
    if disc_optimizer is not None:
        extra_loaders['disc_optimizer'] = disc_optimizer.load_state_dict
    start_epoch, _ = maybe_load_resume(pixel_decoder, optimizer, resume_path, scaler=scaler, extra_state_loaders=extra_loaders or None)

    epochs = int(config['train'].get('epochs', 1))
    for epoch in range(start_epoch, epochs):
        logs = train_stage2_epoch(
            fae=fae,
            pixel_decoder=pixel_decoder,
            backbone=backbone,
            dataloader=dataloader,
            optimizer=optimizer,
            scaler=scaler,
            context=autocast_context,
            device=device,
            image_size=config['pixel_decoder'].get('image_size', 256),
            mode=config['stage2'].get('mode', 'gaussian'),
            discriminator=discriminator,
            disc_optimizer=disc_optimizer,
            perceptual_loss=perceptual_loss,
            noise_std=config['stage2'].get('noise_std', 0.1),
            skip_oom_batches=config["stage2"].get("skip_oom_batches", False)
        )
        print(f"epoch={epoch} " + ' '.join(f"{k}={v:.4f}" for k, v in logs.items()))
        state = {
            'epoch': epoch + 1,
            'model': pixel_decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict() if scaler.is_enabled() else None,
            'config': config,
        }
        if discriminator is not None:
            state['discriminator'] = discriminator.state_dict()
        if disc_optimizer is not None:
            state['disc_optimizer'] = disc_optimizer.state_dict()
        save_checkpoint(output_dir / 'latest.pt', state)


if __name__ == '__main__':
    main()
