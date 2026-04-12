from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)
