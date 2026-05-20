"""Lightweight component-level runtime profiler for diffusion pipelines."""

from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Callable

import csv
import time

import torch


@dataclass
class TimerRecord:
    name: str
    wall_ms: float
    cuda_ms: float | None = None


@dataclass
class ComponentStats:
    name: str
    records: list[TimerRecord] = field(default_factory=list)

    def add(self, wall_ms: float, cuda_ms: float | None = None) :
        self.records.append(
            TimerRecord(name=self.name, wall_ms=float(wall_ms), cuda_ms=None if cuda_ms is None else float(cuda_ms))
        )

    @property
    def calls(self) -> int:
        return len(self.records)

    @property
    def total_wall_ms(self) -> float:
        return sum(r.wall_ms for r in self.records)

    @property
    def total_cuda_ms(self) -> float | None:
        vals = [r.cuda_ms for r in self.records if r.cuda_ms is not None]
        return sum(vals) if vals else None

    def to_summary(self) -> dict[str, Any]:
        wall = [r.wall_ms for r in self.records]
        cuda = [r.cuda_ms for r in self.records if r.cuda_ms is not None]
        out: dict[str, Any] = {
            "calls": self.calls,
            "total_wall_ms": sum(wall) if wall else 0.0,
            "mean_wall_ms": mean(wall) if wall else 0.0,
            "median_wall_ms": median(wall) if wall else 0.0,
            "std_wall_ms": pstdev(wall) if len(wall) > 1 else 0.0,
        }
        if cuda:
            out.update(
                {
                    "total_cuda_ms": sum(cuda),
                    "mean_cuda_ms": mean(cuda),
                    "median_cuda_ms": median(cuda),
                    "std_cuda_ms": pstdev(cuda) if len(cuda) > 1 else 0.0,
                }
            )
        return out


class RuntimeProfiler:
    """Patch selected methods and collect per-call wall/CUDA timings"""

    def __init__(self, device: str | torch.device = "cuda", use_cuda_events: bool = True) :
        self.device = torch.device(device if str(device) != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_cuda_events = bool(use_cuda_events and self.device.type == "cuda" and torch.cuda.is_available())
        self.components: dict[str, ComponentStats] = {}
        self._patches: list[tuple[Any, str, Any]] = []

    def _sync(self) :
        if self.use_cuda_events:
            torch.cuda.synchronize(self.device)

    def _record_call(self, name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self._sync()
        start_event = end_event = None
        if self.use_cuda_events:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            if self.use_cuda_events and end_event is not None and start_event is not None:
                end_event.record()
                torch.cuda.synchronize(self.device)
                cuda_ms = float(start_event.elapsed_time(end_event))
            else:
                cuda_ms = None
            wall_ms = (time.perf_counter() - t0) * 1000.0
            self.components.setdefault(name, ComponentStats(name=name)).add(wall_ms=wall_ms, cuda_ms=cuda_ms)

    def patch_method(self, obj: Any, method_name: str, component_name: str) -> bool:
        if obj is None or not hasattr(obj, method_name):
            return False
        original = getattr(obj, method_name)
        if not callable(original):
            return False

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return self._record_call(component_name, original, *args, **kwargs)

        setattr(obj, method_name, wrapped)
        self._patches.append((obj, method_name, original))
        return True

    def restore(self) :
        for obj, name, original in reversed(self._patches):
            setattr(obj, name, original)
        self._patches.clear()

    def reset(self) :
        self.components.clear()

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {name: stats.to_summary() for name, stats in sorted(self.components.items())}

    def __enter__(self) -> "RuntimeProfiler":
        return self

    def __exit__(self, exc_type, exc, tb) :
        self.restore()


def patch_diffusion_pipeline_components(profiler: RuntimeProfiler, pipe: Any) -> list[str]:
    """Patch common text-to-image pipeline components"""

    patched: list[str] = []
    candidates = [
        (pipe, "encode_prompt", "encode_prompt_total"),
        (getattr(pipe, "text_encoder", None), "forward", "text_encoder_forward"),
        (getattr(pipe, "transformer", None), "forward", "transformer_forward"),
        (getattr(pipe, "unet", None), "forward", "unet_forward"),
        (getattr(pipe, "vae", None), "decode", "vae_decode"),
        (getattr(pipe, "scheduler", None), "step", "scheduler_step"),
        (getattr(pipe, "image_processor", None), "postprocess", "image_postprocess"),
    ]
    for obj, method_name, component_name in candidates:
        if profiler.patch_method(obj, method_name, component_name):
            patched.append(component_name)
    return patched


def summarize_runs(runs: list[dict[str, Any]], warmup_runs: int = 0) -> dict[str, Any]:
    measured = [r for r in runs if not r.get("warmup", False)]
    if not measured:
        measured = runs[int(warmup_runs) :]

    total_ms = [float(r["total_wall_ms"]) for r in measured]
    summary: dict[str, Any] = {
        "num_runs_total": len(runs),
        "num_runs_measured": len(measured),
        "warmup_runs": int(warmup_runs),
        "total_wall_ms": _stats(total_ms),
        "components": {},
    }

    component_names = sorted({name for run in measured for name in run.get("components", {}).keys()})
    total_mean = summary["total_wall_ms"]["mean"] or 0.0
    for name in component_names:
        vals = [float(run["components"].get(name, {}).get("total_wall_ms", 0.0)) for run in measured]
        component_summary = _stats(vals)
        component_summary["mean_percent_of_total_wall"] = (
            100.0 * component_summary["mean"] / total_mean if total_mean else 0.0
        )
        cuda_vals = [
            float(run["components"][name]["total_cuda_ms"])
            for run in measured
            if name in run.get("components", {}) and run["components"][name].get("total_cuda_ms") is not None
        ]
        if cuda_vals:
            component_summary["cuda_ms"] = _stats(cuda_vals)
        summary["components"][name] = component_summary
    return summary


def _stats(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": mean(vals),
        "median": median(vals),
        "std": pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def write_profile_csv(path: str | Path, runs: list[dict[str, Any]], summary: dict[str, Any]) :
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    component_names = sorted({name for run in runs for name in run.get("components", {}).keys()})
    fieldnames = ["run_index", "warmup", "batch_size", "total_wall_ms"]
    fieldnames += [f"{name}_wall_ms" for name in component_names]
    fieldnames += [f"{name}_calls" for name in component_names]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            row = {
                "run_index": run.get("run_index"),
                "warmup": run.get("warmup"),
                "batch_size": run.get("batch_size"),
                "total_wall_ms": run.get("total_wall_ms"),
            }
            for name in component_names:
                stats = run.get("components", {}).get(name, {})
                row[f"{name}_wall_ms"] = stats.get("total_wall_ms", 0.0)
                row[f"{name}_calls"] = stats.get("calls", 0)
            writer.writerow(row)
