from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset

from typing import Iterator, List, Optional
import random

from datasets import load_dataset


def _extract_prompt(example: dict) -> Optional[str]:
    for k in ("sentence", "sentences", "caption", "captions", "text"):
        if k not in example:
            continue
        v = example[k]
        if isinstance(v, dict) and "raw" in v and isinstance(v["raw"], str):
            return v["raw"].strip()
        if isinstance(v, list) and v and isinstance(v[0], dict) and "raw" in v[0]:
            return v[0]["raw"].strip()
        if isinstance(v, str):
            return v.strip()

    if "raw" in example and isinstance(example["raw"], str):
        return example["raw"].strip()

    return None


def iter_coco_prompts(
    *,
    split: str = "train",
    seed: int = 0,
    max_prompts: Optional[int] = None,
    shuffle_buffer: int = 10_000,
) -> Iterator[str]:
    rng = random.Random(seed)
    ds = load_dataset("bitmind/MS-COCO", split=split, streaming=True)

    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    n = 0
    for ex in ds:
        p = _extract_prompt(ex)
        if not p:
            continue
        yield p
        n += 1
        if max_prompts is not None and n >= max_prompts:
            return


def sample_prompts_list(
    *,
    num_prompts: int,
    split: str = "train",
    seed: int = 0,
    max_scan: int = 200_000,
) -> List[str]:
    out: List[str] = []
    it = iter_coco_prompts(split=split, seed=seed, max_prompts=max_scan)
    for p in it:
        out.append(p)
        if len(out) >= num_prompts:
            break
    if len(out) < num_prompts:
        raise RuntimeError(f"Only collected {len(out)} prompts (< {num_prompts}).")
    return out


@dataclass
class LatentsDataset(Dataset):
    latents: torch.Tensor  # (N,C,H,W)
    labels: torch.Tensor   # (N,)

    def __len__(self) -> int:
        return int(self.latents.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.latents[idx], self.labels[idx]

def save_latents(path: Path, latents: torch.Tensor, labels: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"latents": latents.cpu(), "labels": labels.cpu()}, str(path))

def load_latents(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    obj = torch.load(str(path), map_location="cpu")
    return obj["latents"], obj["labels"]
