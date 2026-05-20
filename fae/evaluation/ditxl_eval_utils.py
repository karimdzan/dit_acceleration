"""Utility functions for DiT-XL token-merging evaluation scripts."""

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_yaml_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml
    except Exception as exc:
        raise ImportError("PyYAML is required to read YAML configs. Install with `pip install pyyaml`.") from exc
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping at the top level.")
    return data


def get_nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cur: Any = config
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload: Any) :
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def read_int_labels(path: str | Path) -> list[int]:
    labels: list[int] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            first = stripped.replace(",", " ").replace("\t", " ").split()[0]
            try:
                label = int(first)
            except ValueError as exc:
                raise ValueError(f"Could not parse integer class label on line {line_no} of {path!s}: {line!r}") from exc
            if not 0 <= label < 1000:
                raise ValueError(f"ImageNet class label must be in [0, 999], got {label} on line {line_no}.")
            labels.append(label)
    return labels


def make_class_labels(
    num_samples: int,
    mode: str = "random",
    seed: int = 0,
    labels_file: str | Path | None = None,
) -> list[int]:
    """Create class labels for class-conditional DiT generation.

    Modes:
        random: deterministic uniform random labels in [0, 999].
        round_robin: 0, 1, ..., 999, 0, 1, ...
        labels_file: read integer labels from a file and truncate/repeat to length.
    """

    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}.")
    mode = mode.lower()
    if mode == "random":
        rng = np.random.default_rng(seed)
        return [int(x) for x in rng.integers(0, 1000, size=num_samples)]
    if mode == "round_robin":
        return [i % 1000 for i in range(num_samples)]
    if mode == "labels_file":
        if labels_file is None:
            raise ValueError("labels_file mode requires --labels-file.")
        labels = read_int_labels(labels_file)
        if not labels:
            raise ValueError(f"No labels found in labels file {labels_file!s}.")
        repeats = math.ceil(num_samples / len(labels))
        return (labels * repeats)[:num_samples]
    raise ValueError(f"Unsupported label mode {mode!r}; expected random, round_robin, or labels_file.")


def image_path_for_index(image_dir: str | Path, index: int, label: int) -> Path:
    return Path(image_dir) / f"{index:06d}_class{label:04d}.png"


def make_id_to_label(labels_mapping: Any) -> dict[int, str]:
    """Invert DiTPipeline.labels when available"""

    id_to_label: dict[int, str] = {}
    if isinstance(labels_mapping, dict):
        for name, idx in labels_mapping.items():
            try:
                idx_int = int(idx)
            except Exception:
                continue
            id_to_label.setdefault(idx_int, str(name))
    return id_to_label


def _load_font(size: int = 16) -> ImageFont.ImageFont:
    for candidate in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def make_pair_image(
    baseline: Image.Image,
    token_merge: Image.Image,
    label: int | None = None,
    label_name: str | None = None,
    baseline_title: str = "baseline",
    token_merge_title: str = "token_merge",
    pad: int = 12,
    header: int = 42,
) -> Image.Image:
    """Create a single side-by-side baseline/token-merge comparison image"""

    left = baseline.convert("RGB")
    right = token_merge.convert("RGB")
    if right.size != left.size:
        right = right.resize(left.size, Image.Resampling.LANCZOS)
    w, h = left.size
    canvas = Image.new("RGB", (2 * w + 3 * pad, h + header + 2 * pad), "white")
    canvas.paste(left, (pad, header + pad))
    canvas.paste(right, (2 * pad + w, header + pad))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(16)
    small = _load_font(13)
    draw.text((pad, pad), baseline_title, fill=(0, 0, 0), font=font)
    draw.text((2 * pad + w, pad), token_merge_title, fill=(0, 0, 0), font=font)
    if label is not None:
        label_text = f"class {label}"
        if label_name:
            label_text += f" · {label_name}"
        draw.text((pad, pad + 20), label_text, fill=(40, 40, 40), font=small)
    return canvas


def make_grid(images: Iterable[Image.Image], columns: int = 4, pad: int = 8, bg: str = "white") -> Image.Image:
    images = [img.convert("RGB") for img in images]
    if not images:
        raise ValueError("Cannot build a grid from zero images.")
    columns = max(1, int(columns))
    w, h = images[0].size
    rows = math.ceil(len(images) / columns)
    canvas = Image.new("RGB", (columns * w + (columns + 1) * pad, rows * h + (rows + 1) * pad), bg)
    for idx, img in enumerate(images):
        if img.size != (w, h):
            img = img.resize((w, h), Image.Resampling.LANCZOS)
        row, col = divmod(idx, columns)
        canvas.paste(img, (pad + col * (w + pad), pad + row * (h + pad)))
    return canvas


def build_side_by_side_outputs(
    baseline_dir: str | Path,
    token_merge_dir: str | Path,
    output_dir: str | Path,
    labels: list[int],
    n: int,
    id_to_label: dict[int, str] | None = None,
    grid_columns: int = 2,
) -> dict[str, Any]:
    """Write individual pair images and one comparison grid"""

    baseline_dir = Path(baseline_dir)
    token_merge_dir = Path(token_merge_dir)
    output_dir = ensure_dir(output_dir)
    pairs_dir = ensure_dir(output_dir / "pairs")
    id_to_label = id_to_label or {}
    n = min(int(n), len(labels))
    pair_paths: list[str] = []
    pair_images: list[Image.Image] = []
    for idx in range(n):
        label = int(labels[idx])
        base_path = image_path_for_index(baseline_dir, idx, label)
        merge_path = image_path_for_index(token_merge_dir, idx, label)
        if not base_path.exists() or not merge_path.exists():
            continue
        pair = make_pair_image(
            Image.open(base_path),
            Image.open(merge_path),
            label=label,
            label_name=id_to_label.get(label),
        )
        pair_path = pairs_dir / f"{idx:06d}_class{label:04d}_baseline_vs_token_merge.png"
        pair.save(pair_path)
        pair_paths.append(str(pair_path))
        pair_images.append(pair)
    grid_path = None
    if pair_images:
        grid_path = output_dir / "grid_baseline_vs_token_merge.png"
        make_grid(pair_images, columns=grid_columns).save(grid_path)
    return {
        "num_pairs_written": len(pair_paths),
        "pair_paths": pair_paths,
        "grid_path": str(grid_path) if grid_path is not None else None,
    }
