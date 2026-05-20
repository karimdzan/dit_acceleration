"""Latency benchmarking via CUDA events."""
import contextlib
import statistics
from dataclasses import dataclass, field

import torch


@dataclass
class LatencyTimer:
    _pending: list[tuple] = field(default_factory=list)
    _warmup_count: int = 0

    @contextlib.contextmanager
    def warmup(self):
        torch.cuda.synchronize()
        yield
        torch.cuda.synchronize()
        self._warmup_count += 1

    @contextlib.contextmanager
    def measure(self):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        self._pending.append((start, end))

    def summary(self) -> dict:
        torch.cuda.synchronize()
        t = [s.elapsed_time(e) for s, e in self._pending]
        if not t:
            return {"n": 0, "warmup_runs": self._warmup_count}
        return {
            "n": len(t),
            "warmup_runs": self._warmup_count,
            "mean_ms": statistics.mean(t),
            "median_ms": statistics.median(t),
            "stdev_ms": statistics.stdev(t) if len(t) > 1 else 0.0,
            "min_ms": min(t),
            "max_ms": max(t),
            "p90_ms": _percentile(t, 90),
            "p99_ms": _percentile(t, 99),
        }


def _percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    idx = (p / 100.0) * (len(values) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(values) - 1)
    frac = idx - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def peak_memory_gb() -> float:
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def reset_peak_memory() -> None:
    torch.cuda.reset_peak_memory_stats()
