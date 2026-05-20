import argparse
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from PIL import Image

try:
    from torchvision.utils import make_grid, save_image
except Exception:
    def make_grid(images: torch.Tensor, nrow: int = 8) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(images.shape)}")
        b, c, h, w = images.shape
        nrow = max(int(nrow), 1)
        ncol = min(nrow, b)
        nrows = (b + ncol - 1) // ncol
        grid = images.new_zeros(c, nrows * h, ncol * w)
        for idx, image in enumerate(images):
            row = idx // ncol
            col = idx % ncol
            grid[:, row * h:(row + 1) * h, col * w:(col + 1) * w] = image
        return grid

    def save_image(image: torch.Tensor, path: str | Path) :
        image = denorm(image)
        if image.ndim == 4:
            image = make_grid(image)
        array = image.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
        Image.fromarray(array).save(path)

from fae.config import load_yaml
from fae.scripts.common import (
    build_backbone_from_config,
    build_device,
    build_fae_from_config,
    get_autocast_context,
    get_train_dtype,
)
from fae.training.common import prepare_backbone_inputs
from fae.utils.pretrained import initialize_rae_from_pretrained


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test reconstruction with a pretrained RAE decoder.")
    parser.add_argument("--config", type=str, required=True, help="Config YAML that defines encoder + RAE.")
    parser.add_argument("--input", type=str, required=True, help="Image path or directory of images.")
    parser.add_argument("--output-dir", type=str, default="recon_test_outputs", help="Where to save reconstructions.")
    parser.add_argument("--autoencoder-ckpt", type=str, default=None, help="Optional full autoencoder checkpoint override.")
    parser.add_argument("--decoder-ckpt", type=str, default=None, help="Optional decoder-only checkpoint override.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap when --input is a directory.")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cpu or cuda:0.")
    parser.add_argument("--dtype", type=str, default=None, choices=["fp32", "fp16", "bf16"], help="Override inference dtype.")
    parser.add_argument("--save-individual", action="store_true", help="Save each input/reconstruction pair separately.")
    return parser.parse_args()


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
    x = x.float()
    if float(x.amin()) < -0.05 or float(x.amax()) > 1.05:
        return ((x.clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0)
    return x.clamp(0.0, 1.0)


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(pred, target, reduction="none").flatten(1).mean(dim=1)
    return -10.0 * torch.log10(torch.clamp(mse, min=1e-10))


def main() :
    args = parse_args()
    config = load_yaml(args.config)
    if args.device is not None:
        config["device"] = args.device
    if args.dtype is not None:
        config.setdefault("train", {})["dtype"] = args.dtype
    if args.autoencoder_ckpt is not None:
        config.setdefault("fae", {}).setdefault("pretrained", {})["autoencoder_checkpoint"] = args.autoencoder_ckpt
    if args.decoder_ckpt is not None:
        config.setdefault("fae", {}).setdefault("pretrained", {})["decoder_path"] = args.decoder_ckpt

    device = build_device(config)
    train_dtype = get_train_dtype(config)

    backbone = build_backbone_from_config(config).to(device)
    autoencoder = build_fae_from_config(config, input_dim=backbone.output_dim).to(device)
    load_info = initialize_rae_from_pretrained(autoencoder, config)
    ckpt_path = load_info.get("autoencoder_checkpoint") or load_info.get("decoder_path")
    if not load_info.get("loaded"):
        raise ValueError("Could not infer pretrained RAE weights from config.")

    if train_dtype in (torch.float16, torch.bfloat16):
        autoencoder = autoencoder.to(train_dtype)

    backbone.eval()
    autoencoder.eval()

    input_paths = collect_images(Path(args.input), max_images=args.max_images)
    pil_images, names = load_pil_images(input_paths)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    autocast_context = get_autocast_context(config, device)
    with torch.inference_mode(), autocast_context:
        targets = backbone.build_reconstruction_targets(pil_images, output_size=autoencoder.image_size).to(device)
        backbone_inputs = prepare_backbone_inputs(backbone, pil_images, device)
        features = backbone.forward_features(backbone_inputs).tokens.to(
            device=device,
            dtype=next(autoencoder.parameters()).dtype,
        )
        encoded = autoencoder.encode(features, add_noise=False)
        preds = autoencoder.decode(encoded)

    pred_fp32 = preds.float()
    target_fp32 = targets.float()
    mse_per_image = F.mse_loss(pred_fp32, target_fp32, reduction="none").flatten(1).mean(dim=1)
    l1_per_image = F.l1_loss(pred_fp32, target_fp32, reduction="none").flatten(1).mean(dim=1)
    psnr_per_image = psnr(pred_fp32, target_fp32)

    report_lines = []
    report_lines.append(f"device: {device}")
    report_lines.append(f"dtype: {train_dtype}")
    report_lines.append(f"num_images: {len(input_paths)}")
    report_lines.append(f"checkpoint: {ckpt_path}")
    report_lines.append(f"load_mode: {load_info.get('mode')}")
    report_lines.append(
        "latent stats: "
        f"backbone_mean={features.float().mean().item():.6f}, "
        f"backbone_std={features.float().std().item():.6f}, "
        f"encoded_mean={encoded.float().mean().item():.6f}, "
        f"encoded_std={encoded.float().std().item():.6f}"
    )
    report_lines.append("")
    report_lines.append("per-image metrics:")
    for path, mse_val, l1_val, psnr_val in zip(input_paths, mse_per_image.tolist(), l1_per_image.tolist(), psnr_per_image.tolist()):
        report_lines.append(f"- {path.name}: mse={mse_val:.6f} l1={l1_val:.6f} psnr={psnr_val:.3f}dB")
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


if __name__ == "__main__":
    main()
