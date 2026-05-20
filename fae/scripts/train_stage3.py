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
from fae.utils.checkpoint import load_checkpoint, save_checkpoint
from fae.utils.distributed import barrier, cleanup_distributed, init_distributed_from_env, is_main_process, maybe_set_dataloader_epoch, maybe_wrap_ddp


@hydra.main(version_base=None, config_path=None)
def main(cfg: DictConfig) :
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    resume: str | None = config.pop("resume", None)
    init_distributed_from_env()
    device = build_device(config)

    dataloader = build_dataloader(config)
    backbone = build_backbone_from_config(config).to(device)

    fae = build_fae_from_config(config, input_dim=backbone.output_dim).to(device)
    fae_state = load_checkpoint(config['stage3']['fae_checkpoint'], map_location='cpu')
    fae.load_state_dict(fae_state['model'])
    fae.eval()

    fae_spec = get_fae_latent_spec(config)
    bridge_enabled = bool(config.get('bridge', {}).get('enabled', True))
    generator_spec = fae_spec if not bridge_enabled else None
    if generator_spec is None:
        from fae.generators.common import LatentTensorSpec
        generator_spec = LatentTensorSpec(
            channels=config['generator'].get('in_channels', fae_spec.channels),
            height=config['generator'].get('sample_size', fae_spec.height),
            width=config['generator'].get('sample_size', fae_spec.width),
        )
    generator = build_generator_from_config(config, model_spec=generator_spec).to(device)
    bridge = build_bridge_from_config(config, fae_spec=fae_spec, model_spec=generator.latent_spec()).to(device)

    optim_params = list(bridge.parameters()) + [p for p in generator.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=config['train'].get('lr', 1e-4),
        betas=(0.9, 0.999),
        weight_decay=config['train'].get('weight_decay', 0.05),
    )
    scaler = get_grad_scaler(config)
    autocast_context = get_autocast_context(config, device)
    train_dtype = get_train_dtype(config)
    if train_dtype in (torch.float16, torch.bfloat16):
        fae = fae.to(train_dtype)
        generator = generator.to(train_dtype)
        bridge = bridge.to(train_dtype)

    generator_train = maybe_wrap_ddp(generator, device)
    bridge_train = maybe_wrap_ddp(bridge, device)

    class_conditioner, text_conditioner = maybe_build_conditioners(config, device)
    class_conditioner_train = maybe_wrap_ddp(class_conditioner, device) if class_conditioner is not None else None
    text_conditioner_train = maybe_wrap_ddp(text_conditioner, device) if text_conditioner is not None else None
    cond_params = []
    if class_conditioner_train is not None:
        cond_params += list(class_conditioner_train.parameters())
    if text_conditioner_train is not None:
        cond_params += list(text_conditioner_train.parameters())
    conditioner_optimizer = None
    if cond_params:
        conditioner_optimizer = torch.optim.AdamW(cond_params, lr=config['train'].get('lr', 1e-4))

    output_dir = Path(config['train'].get('output_dir', 'checkpoints/stage3'))
    resume_path = resume or (str(maybe_get_latest_checkpoint(output_dir)) if maybe_get_latest_checkpoint(output_dir) else None)
    extra_loaders = {
        'bridge': bridge.load_state_dict,
    }
    if class_conditioner is not None:
        extra_loaders['class_conditioner'] = class_conditioner.load_state_dict
    if text_conditioner is not None:
        extra_loaders['text_conditioner'] = text_conditioner.load_state_dict
    if conditioner_optimizer is not None:
        extra_loaders['conditioner_optimizer'] = conditioner_optimizer.load_state_dict
    start_epoch, _ = maybe_load_resume(generator, optimizer, resume_path, scaler=scaler, extra_state_loaders=extra_loaders)

    epochs = int(config['train'].get('epochs', 1))
    for epoch in range(start_epoch, epochs):
        maybe_set_dataloader_epoch(dataloader, epoch)
        logs = train_stage3_epoch(
            backend=generator_train,
            fae=fae,
            bridge=bridge_train,
            backbone=backbone,
            dataloader=dataloader,
            optimizer=optimizer,
            scaler=scaler,
            context=autocast_context,
            device=device,
            class_conditioner=class_conditioner_train,
            text_conditioner=text_conditioner_train,
            conditioner_optimizer=conditioner_optimizer,
            bridge_cycle_weight=config['stage3'].get('bridge_cycle_weight', 0.0),
            skip_oom_batches=config['train'].get('skip_oom_batches', True),
        )
        if is_main_process():
            print(f"epoch={epoch} " + ' '.join(f"{k}={v:.4f}" for k, v in logs.items()))
            state = {
                'epoch': epoch + 1,
                'model': generator.state_dict(),
                'bridge': bridge.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict() if scaler.is_enabled() else None,
                'config': config,
            }
            if class_conditioner is not None:
                state['class_conditioner'] = class_conditioner.state_dict()
            if text_conditioner is not None:
                state['text_conditioner'] = text_conditioner.state_dict()
            if conditioner_optimizer is not None:
                state['conditioner_optimizer'] = conditioner_optimizer.state_dict()
            save_checkpoint(output_dir / 'latest.pt', state)
        barrier()
    cleanup_distributed()


if __name__ == '__main__':
    main()
