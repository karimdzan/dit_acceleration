import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageFile, UnidentifiedImageError
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
LOGGER = logging.getLogger(__name__)
_VALID_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


@dataclass
class Sample:
    image: Image.Image
    label: int | None
    caption: str | None
    path: str


class ImageFolderWithOptionalCaptions(Dataset):
    def __init__(
        self,
        root: str | Path,
        captions_jsonl: str | Path | None = None,
        max_samples: int | None = None,
        return_none_on_error: bool = True,
        warn_limit: int = 50,
        load_truncated_images: bool = True,
    ) :
        self.root = Path(root)
        self.records: list[dict[str, Any]] = []
        self.return_none_on_error = bool(return_none_on_error)
        self.warn_limit = int(warn_limit)
        self._warned = 0

        ImageFile.LOAD_TRUNCATED_IMAGES = bool(load_truncated_images)

        captions = {}
        if captions_jsonl is not None:
            with Path(captions_jsonl).open("r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    captions[row["image"]] = row

        classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        class_to_idx = {name: i for i, name in enumerate(classes)}
        found_class_dirs = len(classes) > 0
        if found_class_dirs:
            for class_name in classes:
                class_dir = self.root / class_name
                for path in sorted(class_dir.rglob('*')):
                    if path.suffix.lower() in _VALID_SUFFIXES:
                        rel = str(path.relative_to(self.root))
                        extra = captions.get(rel, {})
                        self.records.append({
                            'path': path,
                            'label': extra.get('label', class_to_idx[class_name]),
                            'caption': extra.get('caption'),
                        })
        else:
            for path in sorted(self.root.rglob('*')):
                if path.suffix.lower() in _VALID_SUFFIXES:
                    rel = str(path.relative_to(self.root))
                    extra = captions.get(rel, {})
                    self.records.append({
                        'path': path,
                        'label': extra.get('label'),
                        'caption': extra.get('caption'),
                    })

        if max_samples is not None:
            self.records = self.records[: int(max_samples)]
        if not self.records:
            raise ValueError(f"No images found under {self.root}.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Sample | None:
        row = self.records[idx]
        try:
            with Image.open(row['path']) as im:
                image = im.convert('RGB')
            return Sample(image=image, label=row.get('label'), caption=row.get('caption'), path=str(row['path']))
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            if self._warned < self.warn_limit:
                LOGGER.warning("Skipping unreadable image %s: %s", row['path'], exc)
                self._warned += 1
            if self.return_none_on_error:
                return None
            raise


def collate_samples(batch: list[Sample | None]) -> dict[str, Any] | None:
    batch = [sample for sample in batch if sample is not None]
    if not batch:
        return None
    return {
        'images': [sample.image for sample in batch],
        'labels': [sample.label for sample in batch],
        'captions': [sample.caption for sample in batch],
        'paths': [sample.path for sample in batch],
    }
