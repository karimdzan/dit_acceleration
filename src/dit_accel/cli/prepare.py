"""Prepare reference data and register clean-fid statistics."""
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from dit_accel.data import (
    build_val_set,
    coco_image_dir,
    prepare_coco_30k,
)
from dit_accel.evaluation.fid import precompute_reference_stats

from ._common import CONFIG_PATH


def _prepare_coco(cfg: DictConfig) -> None:
    local_dir = Path(cfg.local_dir)
    print(f"==> Preparing COCO-30K under {local_dir}")
    prepare_coco_30k(local_dir, force=cfg.force)
    if cfg.register_fid:
        print(f"==> Registering clean-fid stats as '{cfg.register_fid}'")
        from cleanfid import fid as cf
        cf.make_custom_stats(
            cfg.register_fid,
            str(coco_image_dir(local_dir)),
            mode="clean",
            device=cfg.fid_device,
            num_workers=4,
        )


def _prepare_imagenet_val(cfg: DictConfig) -> None:
    if cfg.src_dir is None:
        raise SystemExit("src_dir is required for imagenet_val task.")
    src = Path(cfg.src_dir)
    dst = Path(cfg.local_dir)
    manifest = build_val_set(
        src=src, dst=dst,
        samples_per_class=cfg.samples_per_class,
        mode=cfg.mode, symlink=cfg.symlink,
    )
    print(f"Wrote {manifest['total_images']} images to {dst}")
    if cfg.register_fid:
        from cleanfid import fid as cf
        cf.make_custom_stats(
            cfg.register_fid, str(dst),
            mode="clean", device=cfg.fid_device, num_workers=4,
        )
        print(f"Registered clean-fid stats as '{cfg.register_fid}'")


def _register_fid(cfg: DictConfig) -> None:
    if not cfg.register_fid:
        raise SystemExit("register_fid name is required for fid_stats task.")
    precompute_reference_stats(
        image_dir=cfg.local_dir,
        save_path=Path(f"stats/{cfg.register_fid}.cache"),
        device=cfg.fid_device,
    )


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="prepare")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    task = cfg.task
    if task == "coco30k":
        _prepare_coco(cfg)
    elif task == "imagenet_val":
        _prepare_imagenet_val(cfg)
    elif task == "fid_stats":
        _register_fid(cfg)
    else:
        raise SystemExit(f"unknown prepare task: {task}")


if __name__ == "__main__":
    main()
