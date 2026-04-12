import argparse
import math
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import make_grid, save_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test reconstruction with backbone + FAE + pixel decoder.")
    parser.add_argument("--repo-root", type=str, required=True, help="Path to the repo root that contains the `fae/` package.")
    parser.add_argument("--config", type=str, required=True, help="Stage-2 config YAML.")
    parser.add_argument("--input", type=str, required=True, help="Image path or directory of images.")
    parser.add_argument("--output-dir", type=str, default="recon_test_outputs", help="Where to save reconstructions.")
    parser.add_argument("--fae-ckpt", type=str, default=None, help="Override FAE checkpoint instead of config['stage2']['fae_checkpoint'].")
    parser.add_argument("--pixel-decoder-ckpt", type=str, default=None, help="Override pixel decoder checkpoint instead of config['pixel_decoder']['ckpt'].")
    parser.add_argument(
        "--mode",
        type=str,
        default="finetune",
        choices=["finetune", "gaussian"],
        help="`finetune`: backbone -> FAE -> pixel decoder. `gaussian`: backbone features + noise -> pixel decoder.",
    )
    parser.add_argument("--noise-std", type=float, default=None, help="Noise std for gaussian mode. Defaults to config['stage2']['noise_std'] or 0.1.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap when --input is a directory.")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cpu or cuda:0.")
    parser.add_argument("--dtype", type=str, default=None, choices=["fp32", "fp16", "bf16"], help="Override inference dtype.")
    parser.add_argument("--save-individual", action="store_true", help="Save each input/reconstruction pair separately.")
    return parser.parse_args()


def add_repo_to_path(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def collect_images(input_path: Path, max_images: int | None = None) -> list[Path]:
    valid_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    paths = sorted(p for p in input_path.rglob("*") if p.suffix.lower() in valid_suffixes)
    if max_images is not None:
        paths = paths[: max_images]
    if not paths:
        raise ValueError(f"No supported images found under {input_path}")
    return paths


def load_pil_images(paths: Iterable[Path]) -> tuple[list[Image.Image], list[str]]:
    images: list[Image.Image] = []
    names: list[str] = []
    for path in paths:
        with Image.open(path) as im:
            images.append(im.convert("RGB"))
        names.append(path.stem)
    return images, names


def denorm(x: torch.Tensor) -> torch.Tensor:
    return ((x.clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(pred, target, reduction="none").flatten(1).mean(dim=1)
    return -10.0 * torch.log10(torch.clamp(mse, min=1e-10))


if __name__ == "__main__":
    args = parse_args()
    repo_root = Path(args.repo_root)
    add_repo_to_path(repo_root)

    from fae.config import load_yaml
    from fae.data.transforms import TargetImageTransform
    from fae.training.common import prepare_backbone_inputs
    from fae.scripts.common import (
        build_backbone_from_config,
        build_device,
        build_fae_from_config,
        build_pixel_decoder_from_config,
        get_autocast_context,
        get_train_dtype,
    )
    from fae.utils.checkpoint import load_checkpoint

    config = load_yaml(args.config)
    if args.device is not None:
        config["device"] = args.device
    if args.dtype is not None:
        config.setdefault("train", {})["dtype"] = args.dtype
    if args.fae_ckpt is not None:
        config.setdefault("stage2", {})["fae_checkpoint"] = args.fae_ckpt
    if args.pixel_decoder_ckpt is not None:
        config.setdefault("pixel_decoder", {})["ckpt"] = args.pixel_decoder_ckpt

    device = build_device(config)
    train_dtype = get_train_dtype(config)
    image_size = int(config["pixel_decoder"].get("image_size", 256))
    noise_std = args.noise_std
    if noise_std is None:
        noise_std = float(config.get("stage2", {}).get("noise_std", 0.1))

    backbone = build_backbone_from_config(config).to(device)
    fae = build_fae_from_config(config, input_dim=backbone.output_dim).to(device)
    fae_state = load_checkpoint(config["stage2"]["fae_checkpoint"], map_location="cpu")
    fae.load_state_dict(fae_state["model"])

    pixel_decoder = build_pixel_decoder_from_config(config, input_dim=backbone.output_dim).to(device)

    if train_dtype in (torch.float16, torch.bfloat16):
        fae = fae.to(train_dtype)
        pixel_decoder = pixel_decoder.to(train_dtype)

    backbone.eval()
    fae.eval()
    pixel_decoder.eval()

    input_paths = collect_images(Path(args.input), max_images=args.max_images)
    pil_images, names = load_pil_images(input_paths)
    target_transform = TargetImageTransform(image_size)
    targets = target_transform(pil_images).to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    autocast_context = get_autocast_context(config, device)
    with torch.inference_mode(), autocast_context:
        backbone_inputs = prepare_backbone_inputs(backbone, pil_images, device)
        features = backbone.forward_features(backbone_inputs).tokens.to(
            device=device,
            dtype=next(fae.parameters()).dtype,
        )

        if args.mode == "gaussian":
            decoder_input = features + noise_std * torch.randn_like(features)
            reconstructed_features = None
            z = None
        else:
            fae_out = fae(features)
            z = fae_out.z
            reconstructed_features = fae_out.reconstructed_features
            decoder_input = reconstructed_features

        preds = pixel_decoder(decoder_input)

    pred_fp32 = preds.float()
    target_fp32 = targets.float()
    mse_per_image = F.mse_loss(pred_fp32, target_fp32, reduction="none").flatten(1).mean(dim=1)
    l1_per_image = F.l1_loss(pred_fp32, target_fp32, reduction="none").flatten(1).mean(dim=1)
    psnr_per_image = psnr(pred_fp32, target_fp32)

    report_lines = []
    report_lines.append(f"mode: {args.mode}")
    report_lines.append(f"device: {device}")
    report_lines.append(f"dtype: {train_dtype}")
    report_lines.append(f"num_images: {len(input_paths)}")
    report_lines.append(f"image_size: {image_size}")
    report_lines.append(f"fae_checkpoint: {config['stage2']['fae_checkpoint']}")
    report_lines.append(f"pixel_decoder_checkpoint: {config['pixel_decoder'].get('ckpt')}")
    report_lines.append("")

    if reconstructed_features is not None:
        report_lines.append(
            "feature stats: "
            f"backbone_mean={features.float().mean().item():.6f}, "
            f"backbone_std={features.float().std().item():.6f}, "
            f"fae_mean={reconstructed_features.float().mean().item():.6f}, "
            f"fae_std={reconstructed_features.float().std().item():.6f}"
        )
        if z is not None:
            report_lines.append(
                f"latent stats: z_mean={z.float().mean().item():.6f}, z_std={z.float().std().item():.6f}"
            )
        report_lines.append("")

    report_lines.append("per-image metrics:")
    for path, mse_val, l1_val, psnr_val in zip(input_paths, mse_per_image.tolist(), l1_per_image.tolist(), psnr_per_image.tolist()):
        report_lines.append(
            f"- {path.name}: mse={mse_val:.6f} l1={l1_val:.6f} psnr={psnr_val:.3f}dB"
        )

    report_lines.append("")
    report_lines.append(
        "means: "
        f"mse={mse_per_image.mean().item():.6f} "
        f"l1={l1_per_image.mean().item():.6f} "
        f"psnr={psnr_per_image.mean().item():.3f}dB"
    )

    (output_dir / "metrics.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    vis_target = denorm(targets.detach().cpu())
    vis_pred = denorm(preds.detach().cpu())

    if args.save_individual:
        for i, name in enumerate(names):
            pair = torch.stack([vis_target[i], vis_pred[i]], dim=0)
            grid = make_grid(pair, nrow=2)
            save_image(grid, output_dir / f"{name}_pair.png")
            save_image(vis_pred[i], output_dir / f"{name}_recon.png")

    rows = []
    for i in range(len(input_paths)):
        rows.append(vis_target[i])
        rows.append(vis_pred[i])
    grid = make_grid(torch.stack(rows, dim=0), nrow=2)
    save_image(grid, output_dir / "reconstruction_grid.png")

    print("Saved:")
    print(f"  grid:    {output_dir / 'reconstruction_grid.png'}")
    print(f"  metrics: {output_dir / 'metrics.txt'}")
    if args.save_individual:
        print(f"  per-image pairs and reconstructions under: {output_dir}")
