from pathlib import Path
from typing import Any

import torch


def _is_tensor_state_dict(obj: Any) -> bool:
    return isinstance(obj, dict) and bool(obj) and all(isinstance(k, str) for k in obj.keys()) and any(torch.is_tensor(v) for v in obj.values())


def save_checkpoint(path: str | Path, state: dict[str, Any]) :
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Loading .safetensors checkpoints requires the `safetensors` package. "
                "Install it with `pip install safetensors`."
            ) from exc
        device = map_location
        if isinstance(device, torch.device):
            device = str(device)
        if device is None:
            device = "cpu"
        return load_file(str(path), device=device)
    return torch.load(path, map_location=map_location)


def extract_model_state_dict(state: dict[str, Any], *, prefer_ema: bool = True) -> dict[str, Any]:
    if _is_tensor_state_dict(state):
        return state
    if not isinstance(state, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(state)!r}")
    ordered_keys = []
    if prefer_ema:
        ordered_keys.extend(["ema_model", "model_ema"])
    ordered_keys.extend(["model", "state_dict", "module"])
    for key in ordered_keys:
        value = state.get(key)
        if _is_tensor_state_dict(value):
            return value
    raise KeyError(
        "Could not extract a model state dict from checkpoint. "
        f"Available top-level keys: {sorted(state.keys())}"
    )
