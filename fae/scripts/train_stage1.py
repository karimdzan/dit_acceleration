from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

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
from fae.training.common import build_lr_scheduler
from fae.utils.checkpoint import save_checkpoint
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

    scaler = get_grad_scaler(config)
    autocast_context = get_autocast_context(config, device)
    train_dtype = get_train_dtype(config)
    if train_dtype in (torch.float16, torch.bfloat16):
        fae = fae.to(train_dtype)

    fae_train = maybe_wrap_ddp(fae, device)

    optimizer = torch.optim.AdamW(
        fae_train.parameters(),
        lr=config['train'].get('lr', 1e-4),
        betas=(0.9, 0.999),
        weight_decay=config['train'].get('weight_decay', 0.05),
    )
    scheduler = build_lr_scheduler(optimizer, config['train'], steps_per_epoch=len(dataloader))

    output_dir = Path(config['train'].get('output_dir', 'checkpoints/stage1'))
    resume_path = resume or (str(maybe_get_latest_checkpoint(output_dir)) if maybe_get_latest_checkpoint(output_dir) else None)
    extra_loaders = {'scheduler': scheduler.load_state_dict} if scheduler is not None else None
    start_epoch, _ = maybe_load_resume(fae, optimizer, resume_path, scaler=scaler, extra_state_loaders=extra_loaders)

    epochs = int(config['train'].get('epochs', 1))
    for epoch in range(start_epoch, epochs):
        maybe_set_dataloader_epoch(dataloader, epoch)
        logs = train_stage1_epoch(
            fae_train,
            backbone,
            dataloader,
            optimizer,
            scaler=scaler,
            context=autocast_context,
            device=device,
            lr_scheduler=scheduler,
        )
        if is_main_process():
            print(f"epoch={epoch} " + ' '.join(f"{k}={v:.4f}" for k, v in logs.items()))
            save_checkpoint(output_dir / 'latest.pt', {
                'epoch': epoch + 1,
                'model': fae.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler is not None else None,
                'scaler': scaler.state_dict() if scaler.is_enabled() else None,
                'config': config,
            })
        barrier()
    cleanup_distributed()


if __name__ == '__main__':
    main()
