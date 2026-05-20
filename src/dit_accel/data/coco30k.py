"""MS-COCO 2014 validation 30K subset for T2I evaluation."""
import json
from pathlib import Path


COCO_30K_REPO = "sayakpaul/coco-30-val-2014"
COCO_30K_REVISION: str | None = None


def coco_image_dir(local_dir: str | Path) -> Path:
    return Path(local_dir) / "images"


def coco_captions_path(local_dir: str | Path) -> Path:
    return Path(local_dir) / "captions.json"


def prepare_coco_30k(local_dir: str | Path, force: bool = False) -> None:
    local_dir = Path(local_dir)
    img_dir = coco_image_dir(local_dir)
    caps_path = coco_captions_path(local_dir)

    if not force and caps_path.exists() and img_dir.exists():
        existing = sorted(img_dir.glob("*.jpg"))
        if len(existing) == 30000:
            return

    img_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    ds = load_dataset(
        COCO_30K_REPO,
        split="train",
        revision=COCO_30K_REVISION,
    )

    n = len(ds)
    if n != 30000:
        print(f"warning: expected 30000 rows in {COCO_30K_REPO}, got {n}")

    captions: list[str] = []
    for i in range(n):
        row = ds[i]
        if "image" not in row or "caption" not in row:
            raise KeyError(
                f"Unexpected schema in {COCO_30K_REPO}: row keys = {list(row.keys())}."
            )
        img = row["image"]
        cap = row["caption"]
        out_path = img_dir / f"{i:08d}.jpg"
        if not out_path.exists():
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(out_path, format="JPEG", quality=95)
        captions.append(cap)
        if (i + 1) % 1000 == 0:
            print(f"  materialized {i + 1}/{n}")

    with open(caps_path, "w") as f:
        json.dump(captions, f)

    print(f"COCO-30K ready at {local_dir}")


def load_coco_30k_captions(local_dir: str | Path) -> list[str]:
    caps_path = coco_captions_path(local_dir)
    if not caps_path.exists():
        raise FileNotFoundError(f"{caps_path} missing. Run prepare task first.")
    with open(caps_path) as f:
        captions = json.load(f)
    if not isinstance(captions, list) or len(captions) == 0:
        raise ValueError(f"captions.json at {caps_path} is empty or malformed")
    return captions
