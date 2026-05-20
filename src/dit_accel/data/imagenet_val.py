"""Build a frozen ImageNet validation subset from a class-subdir source."""
import json
import os
import shutil
import sys
from pathlib import Path


VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".JPEG", ".JPG", ".PNG", ".WEBP"}


def collect_class_dirs(src: Path) -> list[Path]:
    if not src.is_dir():
        raise SystemExit(f"src {src} is not a directory")
    dirs = sorted(d for d in src.iterdir() if d.is_dir())
    if not dirs:
        raise SystemExit(f"No subdirectories found in {src}.")
    return dirs


def collect_images(class_dir: Path) -> list[Path]:
    return sorted(
        f for f in class_dir.iterdir()
        if f.is_file() and f.suffix in VALID_EXTS
    )


def place(src_file: Path, dst_file: Path, symlink: bool) -> None:
    if dst_file.exists() or dst_file.is_symlink():
        return
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    if symlink:
        os.symlink(src_file.resolve(), dst_file)
    else:
        shutil.copy2(src_file, dst_file)


def build_val_set(
    src: Path,
    dst: Path,
    samples_per_class: int = 0,
    mode: str = "flat",
    symlink: bool = True,
) -> dict:
    dst.mkdir(parents=True, exist_ok=True)
    class_dirs = collect_class_dirs(src)
    print(f"Found {len(class_dirs)} class subdirectories under {src}")

    emit: list[tuple[str, Path]] = []
    per_class_counts: dict[str, int] = {}
    for class_dir in class_dirs:
        name = class_dir.name
        files = collect_images(class_dir)
        if not files:
            print(f"  warning: {name} has no images, skipping", file=sys.stderr)
            continue
        if samples_per_class > 0:
            files = files[:samples_per_class]
        per_class_counts[name] = len(files)
        emit.extend((name, f) for f in files)

    if not emit:
        raise SystemExit("Nothing to emit; check src and image extensions.")

    print(f"Emitting {len(emit)} images ({'symlinks' if symlink else 'copies'}, mode={mode})")

    if mode == "flat":
        for i, (name, src_file) in enumerate(emit):
            ext = src_file.suffix
            target = dst / f"{i:08d}_{name}{ext}"
            place(src_file, target, symlink)
    else:
        for name, src_file in emit:
            target = dst / name / src_file.name
            place(src_file, target, symlink)

    manifest = {
        "src": str(src),
        "dst": str(dst),
        "mode": mode,
        "symlink": symlink,
        "samples_per_class": samples_per_class,
        "num_classes": len(per_class_counts),
        "total_images": len(emit),
        "per_class_counts": per_class_counts,
    }
    with open(dst / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest
