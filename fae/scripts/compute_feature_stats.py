from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from fae.scripts.common import build_backbone_from_config, build_dataloader, build_device
from fae.training.common import prepare_backbone_inputs
from fae.utils.checkpoint import save_checkpoint


@hydra.main(version_base=None, config_path=None)
def main(cfg: DictConfig) :
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    feat_cfg = config.get('compute_features', {})
    output_path = Path(feat_cfg.get('output', 'checkpoints/latent_stats/latent_stats.pt'))
    max_batches: int | None = feat_cfg.get('max_batches') or None
    channel_only: bool = bool(feat_cfg.get('channel_only', False))

    device = build_device(config)
    dataloader = build_dataloader(config)
    backbone = build_backbone_from_config(config).to(device)
    backbone.eval()

    running_sum = None
    running_sq_sum = None
    running_count = 0
    num_images = 0
    num_batches = 0
    spatial_shape = None

    for batch in tqdm(dataloader, desc="latent_stats"):
        images = batch["images"]
        backbone_inputs = prepare_backbone_inputs(backbone, images, device)
        features = backbone.forward_features(backbone_inputs).tokens.float()
        b, n, c = features.shape
        h = w = int(n ** 0.5)
        feature_grid = features.transpose(1, 2).reshape(b, c, h, w)
        if channel_only:
            batch_sum = feature_grid.sum(dim=(0, 2, 3)).cpu()
            batch_sq_sum = feature_grid.square().sum(dim=(0, 2, 3)).cpu()
            batch_count = int(b * h * w)
        else:
            batch_sum = feature_grid.sum(dim=0).cpu()
            batch_sq_sum = feature_grid.square().sum(dim=0).cpu()
            batch_count = int(b)
            spatial_shape = (c, h, w)

        if running_sum is None:
            running_sum = torch.zeros_like(batch_sum)
            running_sq_sum = torch.zeros_like(batch_sq_sum)

        running_sum += batch_sum
        running_sq_sum += batch_sq_sum
        running_count += batch_count
        num_images += int(b)
        num_batches += 1

        if max_batches is not None and num_batches >= max_batches:
            break

    if running_count == 0 or running_sum is None or running_sq_sum is None:
        raise RuntimeError("No features were processed. Check your dataset/config.")

    mean = running_sum / running_count
    var = (running_sq_sum / running_count) - mean.square()
    std = var.clamp(min=1e-12).sqrt()

    save_checkpoint(
        output_path,
        {
            "mean": mean,
            "var": var,
            "std": std,
            "count": running_count,
            "num_images": num_images,
            "num_batches": num_batches,
            "shape": spatial_shape if spatial_shape is not None else (int(mean.numel()),),
            "encoder": dict(config["encoder"]),
            "data_root": config["data"]["root"],
        },
    )
    print(f"saved latent stats to {output_path}")
    print(f"shape={tuple(mean.shape)} count={running_count} images={num_images} batches={num_batches}")
    print(f"mean_abs={mean.abs().mean().item():.6f} std_mean={std.mean().item():.6f} std_min={std.min().item():.6f}")


if __name__ == "__main__":
    main()
