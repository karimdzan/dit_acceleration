from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from fae.scripts.common import (
    build_backbone_from_config,
    build_bridge_from_config,
    build_dataloader,
    build_device,
    build_fae_from_config,
    build_generator_from_config,
    get_autocast_context,
    get_fae_latent_spec,
    get_grad_scaler,
    get_train_dtype,
    maybe_build_conditioners,
    maybe_get_latest_checkpoint,
    maybe_load_resume,
)
from fae.training import train_stage3_epoch
from fae.training.common import ExponentialMovingAverage, accumulation_steps_from_config, build_lr_scheduler
from fae.utils.checkpoint import load_checkpoint, save_checkpoint
from fae.utils.pretrained import initialize_rae_from_pretrained


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
    autoencoder.eval()

    ae_spec = get_fae_latent_spec(config, input_dim=backbone.output_dim, backbone=backbone)
    bridge_enabled = bool(config.get('bridge', {}).get('enabled', True))
    generator_spec = ae_spec if not bridge_enabled else None
    if generator_spec is None:
        from fae.generators.common import LatentTensorSpec
        generator_spec = LatentTensorSpec(
            channels=config['generator'].get('in_channels', ae_spec.channels),
            height=config['generator'].get('sample_size', ae_spec.height),
            width=config['generator'].get('sample_size', ae_spec.width),
        )
    generator = build_generator_from_config(config, model_spec=generator_spec).to(device)
    bridge = build_bridge_from_config(config, fae_spec=ae_spec, model_spec=generator.latent_spec()).to(device)

    optim_params = list(bridge.parameters()) + [p for p in generator.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=config['train'].get('lr', 1e-4),
        betas=tuple(config['train'].get('betas', (0.9, 0.95))),
        weight_decay=config['train'].get('weight_decay', 0.0),
    )
    scaler = get_grad_scaler(config)
    autocast_context = get_autocast_context(config, device)
    train_dtype = get_train_dtype(config)
    if train_dtype in (torch.float16, torch.bfloat16):
        autoencoder = autoencoder.to(train_dtype)
        generator = generator.to(train_dtype)
        bridge = bridge.to(train_dtype)

    class_conditioner, text_conditioner = maybe_build_conditioners(config, device)
    cond_params = []
    if class_conditioner is not None:
        cond_params += list(class_conditioner.parameters())
    if text_conditioner is not None:
        cond_params += list(text_conditioner.parameters())
    conditioner_optimizer = None
    if cond_params:
        conditioner_optimizer = torch.optim.AdamW(cond_params, lr=config['train'].get('lr', 1e-4))

    output_dir = Path(config['train'].get('output_dir', 'checkpoints/stage3'))
    ema_decay = float(config['train'].get('ema_decay', 0.0) or 0.0)
    ema = ExponentialMovingAverage(generator, decay=ema_decay) if ema_decay > 0 else None
    lr_scheduler = build_lr_scheduler(
        optimizer, config['train'],
        steps_per_epoch=max(len(dataloader) // accumulation_steps_from_config(config['train']), 1),
    )

    resume_path = resume or (str(maybe_get_latest_checkpoint(output_dir)) if maybe_get_latest_checkpoint(output_dir) else None)
    extra_loaders = {'bridge': bridge.load_state_dict}
    if class_conditioner is not None:
        extra_loaders['class_conditioner'] = class_conditioner.load_state_dict
    if text_conditioner is not None:
        extra_loaders['text_conditioner'] = text_conditioner.load_state_dict
    if conditioner_optimizer is not None:
        extra_loaders['conditioner_optimizer'] = conditioner_optimizer.load_state_dict
    if ema is not None:
        extra_loaders['ema'] = ema.load_state_dict
    start_epoch, _ = maybe_load_resume(generator, optimizer, resume_path, scaler=scaler, extra_state_loaders=extra_loaders)

    epochs = int(config['train'].get('epochs', 1))
    accumulation_steps = accumulation_steps_from_config(config['train'])
    grad_clip = config['train'].get('clip_grad', config['train'].get('grad_clip'))
    for epoch in range(start_epoch, epochs):
        logs = train_stage3_epoch(
            backend=generator,
            fae=autoencoder,
            bridge=bridge,
            backbone=backbone,
            dataloader=dataloader,
            optimizer=optimizer,
            scaler=scaler,
            context=autocast_context,
            device=device,
            class_conditioner=class_conditioner,
            text_conditioner=text_conditioner,
            conditioner_optimizer=conditioner_optimizer,
            bridge_cycle_weight=config.get('stage3', {}).get('bridge_cycle_weight', 0.0),
            skip_oom_batches=config['train'].get('skip_oom_batches', True),
            accumulation_steps=accumulation_steps,
            grad_clip=grad_clip,
            lr_scheduler=lr_scheduler,
            ema=ema,
        )
        print(f"epoch={epoch} " + ' '.join(f"{k}={v:.4f}" for k, v in logs.items()))
        state = {
            'epoch': epoch + 1,
            'model': generator.state_dict(),
            'bridge': bridge.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict() if scaler.is_enabled() else None,
            'config': config,
        }
        if ema is not None:
            state['ema'] = ema.state_dict()
            state['ema_model'] = ema.shadow.state_dict()
        if class_conditioner is not None:
            state['class_conditioner'] = class_conditioner.state_dict()
        if text_conditioner is not None:
            state['text_conditioner'] = text_conditioner.state_dict()
        if conditioner_optimizer is not None:
            state['conditioner_optimizer'] = conditioner_optimizer.state_dict()
        save_checkpoint(output_dir / 'latest.pt', state)


if __name__ == '__main__':
    main()
