import copy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at {path}, got {type(data)!r}.")
    return data


def deep_update(base: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in other.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out
