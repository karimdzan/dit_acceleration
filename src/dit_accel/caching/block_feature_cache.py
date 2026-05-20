"""Block-level residual feature cache for diffusers DiT-style transformers."""
import torch
from typing import Any, Callable


_EPS = 1e-8


class BlockFeatureCacheStore:
    def __init__(self, schedule: dict | None = None):
        self.schedule: dict[tuple[int, int], bool] = schedule or {}
        self.step_idx: int = 0
        self.residuals: dict[int, torch.Tensor] = {}
        self.deltas: dict[tuple[int, int], float] = {}
        self.calibration_mode: bool = False
        self.hits: int = 0
        self.misses: int = 0

    def clear(self) -> None:
        """Full reset, including cumulative hit/miss counters."""
        self.residuals.clear()
        self.step_idx = 0
        self.deltas.clear()
        self.hits = 0
        self.misses = 0

    def clear_for_generation(self) -> None:
        """Per-batch reset; preserves cumulative hit/miss counters."""
        self.residuals.clear()
        self.step_idx = 0
        self.deltas.clear()

    def advance_step(self) -> None:
        self.step_idx += 1

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_block_evals": total,
            "hit_rate": (self.hits / total) if total else 0.0,
        }

    def should_skip(self, block_idx: int) -> bool:
        if self.step_idx == 0:
            return False
        if self.calibration_mode:
            return False
        return bool(self.schedule.get((block_idx, self.step_idx), False))


def _make_wrapped_forward(
    orig_forward: Callable, block_idx: int, store: BlockFeatureCacheStore
) -> Callable:
    def wrapped(hidden_states: torch.Tensor, *args: Any, **kwargs: Any):
        skip = store.should_skip(block_idx)
        cached = store.residuals.get(block_idx)

        if skip and cached is not None and cached.shape == hidden_states.shape:
            store.hits += 1
            return hidden_states + cached

        out = orig_forward(hidden_states, *args, **kwargs)
        store.misses += 1

        if isinstance(out, tuple):
            new_hidden = out[0]
            rest = out[1:]
        else:
            new_hidden = out
            rest = ()

        if new_hidden.shape == hidden_states.shape:
            with torch.no_grad():
                residual = (new_hidden - hidden_states).detach()
                if store.calibration_mode and block_idx in store.residuals:
                    prev = store.residuals[block_idx]
                    if prev.shape == residual.shape:
                        num = (residual - prev).norm()
                        den = residual.norm() + _EPS
                        store.deltas[(block_idx, store.step_idx)] = float((num / den).item())
                store.residuals[block_idx] = residual

        if rest:
            return (new_hidden,) + rest
        return new_hidden

    return wrapped


def install_block_feature_cache(
    pipe, schedule: dict | None = None
) -> BlockFeatureCacheStore:
    store = BlockFeatureCacheStore(schedule=schedule)
    pipe._dit_accel_block_cache = store

    transformer = pipe.transformer
    if not hasattr(transformer, "transformer_blocks"):
        raise AttributeError(
            "pipe.transformer has no transformer_blocks attribute."
        )

    for i, block in enumerate(transformer.transformer_blocks):
        if hasattr(block, "_orig_forward_for_cache"):
            orig_forward = block._orig_forward_for_cache
        else:
            orig_forward = block.forward
            block._orig_forward_for_cache = orig_forward
        block.forward = _make_wrapped_forward(orig_forward, i, store)

    sched = pipe.scheduler
    if not hasattr(sched, "_dit_accel_orig_step"):
        sched._dit_accel_orig_step = sched.step

        def step_with_advance(*args, **kwargs):
            result = sched._dit_accel_orig_step(*args, **kwargs)
            store.advance_step()
            return result

        sched.step = step_with_advance

    return store


def uninstall_block_feature_cache(pipe) -> None:
    transformer = pipe.transformer
    for block in transformer.transformer_blocks:
        if hasattr(block, "_orig_forward_for_cache"):
            block.forward = block._orig_forward_for_cache
            del block._orig_forward_for_cache
    sched = pipe.scheduler
    if hasattr(sched, "_dit_accel_orig_step"):
        sched.step = sched._dit_accel_orig_step
        del sched._dit_accel_orig_step
    if hasattr(pipe, "_dit_accel_block_cache"):
        del pipe._dit_accel_block_cache


def build_greedy_schedule(
    deltas_by_step: dict[int, dict[int, dict]],
    k_per_step: int,
) -> dict[tuple[int, int], bool]:
    """Mark the k_per_step blocks with smallest mean delta at each step >= 1."""
    schedule: dict[tuple[int, int], bool] = {}
    for step, blocks in deltas_by_step.items():
        step = int(step)
        if step == 0:
            continue
        sorted_blocks = sorted(blocks.items(), key=lambda kv: kv[1]["mean"])
        skip_blocks = [int(b) for b, _ in sorted_blocks[:k_per_step]]
        for b in skip_blocks:
            schedule[(b, step)] = True
    return schedule
