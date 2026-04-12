import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from fae.config import load_yaml
from fae.scripts.common import build_backbone_from_config, build_dataloader, build_device
from fae.training.common import prepare_backbone_inputs
from fae.utils.checkpoint import save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute frozen-backbone feature mean/std for stage-1 normalization.")
    parser.add_argument("--config", type=str, required=True, help="Training config that defines encoder/data/train batch size.")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Where to save the feature stats .pt file.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap on the number of dataloader batches to scan.",
    )
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    config = load_yaml(args.config)
    device = build_device(config)

    dataloader = build_dataloader(config)
    backbone = build_backbone_from_config(config).to(device)
    backbone.eval()

    running_sum = None
    running_sq_sum = None
    running_count = 0
    num_images = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="feature_stats"):
        images = batch["images"]
        backbone_inputs = prepare_backbone_inputs(backbone, images, device)
        features = backbone.forward_features(backbone_inputs).tokens.float()

        batch_sum = features.sum(dim=(0, 1)).cpu()
        batch_sq_sum = features.square().sum(dim=(0, 1)).cpu()
        batch_count = int(features.shape[0] * features.shape[1])

        if running_sum is None:
            running_sum = torch.zeros_like(batch_sum)
            running_sq_sum = torch.zeros_like(batch_sq_sum)

        running_sum += batch_sum
        running_sq_sum += batch_sq_sum
        running_count += batch_count
        num_images += int(features.shape[0])
        num_batches += 1

        if args.max_batches is not None and num_batches >= args.max_batches:
            break

    if running_count == 0 or running_sum is None or running_sq_sum is None:
        raise RuntimeError("No features were processed. Check your dataset/config.")

    mean = running_sum / running_count
    var = (running_sq_sum / running_count) - mean.square()
    std = var.clamp(min=1e-12).sqrt()

    output_path = Path(args.output)
    save_checkpoint(
        output_path,
        {
            "mean": mean,
            "std": std,
            "count": running_count,
            "num_images": num_images,
            "num_batches": num_batches,
            "feature_dim": int(mean.numel()),
            "encoder": dict(config["encoder"]),
            "data_root": config["data"]["root"],
        },
    )
    print(f"saved feature stats to {output_path}")
    print(f"feature_dim={mean.numel()} tokens_processed={running_count} images={num_images} batches={num_batches}")
    print(f"mean_abs={mean.abs().mean().item():.6f} std_mean={std.mean().item():.6f} std_min={std.min().item():.6f}")


if __name__ == "__main__":
    main()
