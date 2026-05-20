import argparse
import csv
import json
import random
import shutil
from pathlib import Path

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Build a deterministic ImageNet-1K 256 subsample from class folders")
    p.add_argument("--data-root", default="/tmp/data", help="Root with ImageNet class directories, e.g. /tmp/data/abacus, ...")
    p.add_argument("--out", required=True, help="Output directory for manifest and optional copied images")
    p.add_argument("--samples-per-class", type=int, default=8)
    p.add_argument("--max-classes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--copy", action="store_true", help="Copy images into out/images/<class>/ instead of only writing a manifest")
    p.add_argument("--symlink", action="store_true", help="Symlink images into out/images/<class>/; ignored if --copy is set")
    return p.parse_args()


def main() :
    args = parse_args()
    root = Path(args.data_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    classes = sorted(p for p in root.iterdir() if p.is_dir())[: args.max_classes]
    rng = random.Random(args.seed)
    rows = []
    for label, class_dir in enumerate(classes):
        images = sorted(p for p in class_dir.rglob("*") if p.suffix.lower() in _IMAGE_EXTENSIONS)
        if not images:
            continue
        chosen = images if len(images) <= args.samples_per_class else rng.sample(images, args.samples_per_class)
        for idx, src in enumerate(sorted(chosen)):
            rel_dst = Path("images") / class_dir.name / f"{idx:05d}{src.suffix.lower()}"
            dst = out / rel_dst
            if args.copy or args.symlink:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                if args.copy:
                    shutil.copy2(src, dst)
                else:
                    dst.symlink_to(src)
                path_for_manifest = rel_dst.as_posix()
            else:
                path_for_manifest = str(src)
            rows.append({"path": path_for_manifest, "class_name": class_dir.name, "label": label})

    manifest = out / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "class_name", "label"])
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps({
        "data_root": str(root),
        "classes": len(classes),
        "images": len(rows),
        "samples_per_class": args.samples_per_class,
        "seed": args.seed,
        "mode": "copy" if args.copy else "symlink" if args.symlink else "manifest_only",
    }, indent=2))
    print(f"Wrote {len(rows)} images from {len(classes)} classes to {manifest}")


if __name__ == "__main__":
    main()
