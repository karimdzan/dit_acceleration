"""Compute FID and Inception Score for one or more generated image folders"""

import argparse
from pathlib import Path

from fae.evaluation.ditxl_eval_utils import write_json
from fae.evaluation.image_folder_metrics import calculate_fid_is_with_torch_fidelity


def _parse_fake_dir(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        name, path = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid --fake-dir spec {spec!r}: empty name before '='.")
        return name, Path(path)
    path = Path(spec)
    return path.name, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dir", required=True, help="ImageNet validation image folder, e.g. /data/imagenet/val.")
    parser.add_argument(
        "--fake-dir",
        action="append",
        required=True,
        help="Generated folder. Use name=path to control the metric key. Can be repeated.",
    )
    parser.add_argument("--output-json", default=None, help="Optional path for metrics JSON.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true", help="Force CPU metric computation.")
    parser.add_argument("--kid", action="store_true", help="Also compute KID.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() :
    args = parse_args()
    results = {}
    for spec in args.fake_dir:
        name, fake_dir = _parse_fake_dir(spec)
        metrics = calculate_fid_is_with_torch_fidelity(
            real_dir=args.real_dir,
            fake_dir=fake_dir,
            batch_size=args.batch_size,
            cuda=not args.cpu,
            num_workers=args.num_workers,
            verbose=not args.quiet,
            compute_kid=args.kid,
        )
        results[name] = metrics
        print(f"[{name}] {metrics}")
    if args.output_json:
        write_json(args.output_json, results)
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
