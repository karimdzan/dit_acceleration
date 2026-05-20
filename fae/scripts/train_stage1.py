from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from fae.models import NLayerDiscriminator
from fae.scripts.common import (
    build_backbone_from_config,
    build_dataloader,
    build_device,
    build_fae_from_config,
    get_autocast_context,
    get_grad_scaler,
    get_train_dtype,
    maybe_get_latest_checkpoint,
    maybe_load_resume,
)
from fae.training import train_stage1_epoch
from fae.training.common import ExponentialMovingAverage, accumulation_steps_from_config, build_lr_scheduler
from fae.utils.checkpoint import save_checkpoint
from fae.utils.pretrained import initialize_rae_from_pretrained
from fae.utils.losses import VGGPerceptualLoss


@hydra.main(version_base=None, config_path=None)
def main(cfg: DictConfig) :
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    resume: str | None = config.pop("resume", None)
    device = build_device(config)

    dataloader = build_dataloader(config)
    backbone = build_backbone_from_config(config).to(device)
    autoencoder = build_fae_from_config(config, input_dim=backbone.output_dim).to(device)
    pretrained_info = initialize_rae_from_pretrained(autoencoder, config)
    if pretrained_info.get("loaded"):
        print(
            f"Initialized RAE from pretrained {pretrained_info['mode']} weights: "
            f"{pretrained_info.get('autoencoder_checkpoint') or pretrained_info.get('decoder_path')}"
        )

    optimizer = torch.optim.AdamW(
        autoencoder.decoder.parameters(),
        lr=config['train'].get('lr', 1e-4),
        betas=tuple(config['train'].get('betas', (0.9, 0.999))),
        weight_decay=config['train'].get('weight_decay', 0.0),
    )

    stage_cfg = config.get('stage1', {})
    discriminator = None
    disc_optimizer = None
    if stage_cfg.get('use_gan', False):
        discriminator = NLayerDiscriminator().to(device)
        disc_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=stage_cfg.get('disc_lr', config['train'].get('lr', 1e-4)),
            betas=tuple(stage_cfg.get('disc_betas', (0.5, 0.9))),
            weight_decay=0.0,
        )
    perceptual_loss = VGGPerceptualLoss().to(device) if stage_cfg.get('use_lpips', True) else None

    scaler = get_grad_scaler(config)
    autocast_context = get_autocast_context(config, device)
    train_dtype = get_train_dtype(config)
    if train_dtype in (torch.float16, torch.bfloat16):
        autoencoder = autoencoder.to(train_dtype)
        if discriminator is not None:
            discriminator = discriminator.to(train_dtype)

    output_dir = Path(config['train'].get('output_dir', 'checkpoints/stage1'))
    ema_decay = float(config['train'].get('ema_decay', stage_cfg.get('ema_decay', 0.0) or 0.0))
    ema = ExponentialMovingAverage(autoencoder, decay=ema_decay) if ema_decay > 0 else None
    lr_scheduler = build_lr_scheduler(
        optimizer, config['train'],
        steps_per_epoch=max(len(dataloader) // accumulation_steps_from_config(config['train']), 1),
    )

    resume_path = resume or (str(maybe_get_latest_checkpoint(output_dir)) if maybe_get_latest_checkpoint(output_dir) else None)
    extra_loaders = {}
    if discriminator is not None:
        extra_loaders['discriminator'] = discriminator.load_state_dict
    if disc_optimizer is not None:
        extra_loaders['disc_optimizer'] = disc_optimizer.load_state_dict
    if ema is not None:
        extra_loaders['ema'] = ema.load_state_dict
    start_epoch, _ = maybe_load_resume(autoencoder, optimizer, resume_path, scaler=scaler, extra_state_loaders=extra_loaders or None)

    epochs = int(config['train'].get('epochs', 1))
    accumulation_steps = accumulation_steps_from_config(config['train'])
    grad_clip = config['train'].get('clip_grad', config['train'].get('grad_clip'))
    for epoch in range(start_epoch, epochs):
        logs = train_stage1_epoch(
            model=autoencoder,
            backbone=backbone,
            dataloader=dataloader,
            optimizer=optimizer,
            scaler=scaler,
            context=autocast_context,
            device=device,
            discriminator=discriminator,
            disc_optimizer=disc_optimizer,
            perceptual_loss=perceptual_loss,
            current_epoch=epoch,
            perceptual_weight=stage_cfg.get('perceptual_weight', 0.1),
            adversarial_weight=stage_cfg.get('adversarial_weight', 0.01),
            use_adaptive_gan_weight=stage_cfg.get('use_adaptive_gan_weight', True),
            lpips_start_epoch=stage_cfg.get('lpips_start_epoch', 0),
            disc_start_epoch=stage_cfg.get('disc_start_epoch', 6),
            adv_start_epoch=stage_cfg.get('adv_start_epoch', 8),
            skip_oom_batches=config['train'].get('skip_oom_batches', True),
            accumulation_steps=accumulation_steps,
            grad_clip=grad_clip,
            lr_scheduler=lr_scheduler,
            ema=ema,
        )
        print(f"epoch={epoch} " + ' '.join(f"{k}={v:.4f}" for k, v in logs.items()))
        state = {
            'epoch': epoch + 1,
            'model': autoencoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict() if scaler.is_enabled() else None,
            'config': config,
        }
        if ema is not None:
            state['ema'] = ema.state_dict()
            state['ema_model'] = ema.shadow.state_dict()
        if discriminator is not None:
            state['discriminator'] = discriminator.state_dict()
        if disc_optimizer is not None:
            state['disc_optimizer'] = disc_optimizer.state_dict()
        save_checkpoint(output_dir / 'latest.pt', state)


if __name__ == '__main__':
    main()
