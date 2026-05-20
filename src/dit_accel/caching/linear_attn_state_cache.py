"""Linear-attention state cache for Sana / Sana Sprint.

Caches S = phi(K)^T V and z = phi(K)^T 1 per layer, optionally reusing
them at scheduled (layer, step) pairs instead of recomputing.
"""
import torch
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention


_EPS = 1e-15


class StateCacheStore:
    def __init__(self, schedule: dict | None = None, auto_threshold: float = 0.0):
        self.schedule = schedule or {}
        self.auto_threshold = float(auto_threshold)
        self.step_idx = 0
        self._S: dict[int, torch.Tensor] = {}
        self._z: dict[int, torch.Tensor] = {}
        self.hits = 0
        self.misses = 0
        self.deltas: dict[tuple[int, int], float] = {}
        self.calibration_mode = False

    def clear(self) -> None:
        self._S.clear()
        self._z.clear()
        self.step_idx = 0
        self.deltas.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_attn_evals": total,
            "hit_rate": (self.hits / total) if total else 0.0,
        }

    def advance_step(self) -> None:
        self.step_idx += 1

    def should_skip(self, layer_id: int) -> bool:
        if self.step_idx == 0:
            return False
        if (layer_id, self.step_idx) in self.schedule:
            return bool(self.schedule[(layer_id, self.step_idx)])
        return False


class CachedSanaLinearAttnProcessor:
    def __init__(self, layer_id: int, store: StateCacheStore):
        self.layer_id = layer_id
        self.store = store

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        B, N, _ = hidden_states.shape
        H = attn.heads

        q = attn.to_q(hidden_states)
        if attn.norm_q is not None:
            q = attn.norm_q(q)
        D = q.shape[-1] // H
        q = q.view(B, N, H, D).transpose(1, 2)
        phi_q = F.relu(q) + _EPS

        store = self.store
        skip = store.should_skip(self.layer_id)

        if skip and self.layer_id in store._S:
            S = store._S[self.layer_id]
            z = store._z[self.layer_id]
            store.hits += 1
        else:
            store.misses += 1
            k = attn.to_k(encoder_hidden_states)
            v = attn.to_v(encoder_hidden_states)
            if attn.norm_k is not None:
                k = attn.norm_k(k)
            k = k.view(B, N, H, D).transpose(1, 2)
            v = v.view(B, N, H, D).transpose(1, 2)
            phi_k = F.relu(k) + _EPS
            S = torch.einsum("bhnd,bhne->bhde", phi_k, v)
            z = phi_k.sum(dim=2)

            if (
                store.auto_threshold > 0
                and self.layer_id in store._S
                and store.step_idx > 0
            ):
                prev = store._S[self.layer_id]
                if prev.shape == S.shape:
                    rel = (S - prev).norm() / (S.norm() + _EPS)
                    if rel.item() < store.auto_threshold:
                        S = prev
                        z = store._z[self.layer_id]

            if store.calibration_mode and self.layer_id in store._S:
                prev = store._S[self.layer_id]
                if prev.shape == S.shape:
                    with torch.no_grad():
                        rel = ((S - prev).norm() / (S.norm() + _EPS)).item()
                    store.deltas[(self.layer_id, store.step_idx)] = rel

            store._S[self.layer_id] = S
            store._z[self.layer_id] = z

        num = torch.einsum("bhnd,bhde->bhne", phi_q, S)
        den = torch.einsum("bhnd,bhd->bhn", phi_q, z).unsqueeze(-1)
        out = num / (den + _EPS)

        out = out.transpose(1, 2).reshape(B, N, H * D).to(hidden_states.dtype)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out / attn.rescale_output_factor


def install_state_cache(
    pipe,
    schedule: dict | None = None,
    auto_threshold: float = 0.0,
) -> StateCacheStore:
    store = StateCacheStore(schedule=schedule, auto_threshold=auto_threshold)
    pipe._dit_accel_state_cache = store

    sched = pipe.scheduler
    orig_step = sched.step

    def step_with_advance(*args, **kwargs):
        result = orig_step(*args, **kwargs)
        store.advance_step()
        return result

    sched.step = step_with_advance

    layer_id = 0
    for block in pipe.transformer.transformer_blocks:
        if getattr(block, "attn1", None) is None:
            continue
        block.attn1.set_processor(CachedSanaLinearAttnProcessor(layer_id, store))
        layer_id += 1

    return store


def build_greedy_schedule(deltas_path, k_per_step: int) -> dict:
    """Mark the k_per_step layers with smallest mean delta at each step >= 1."""
    payload = torch.load(deltas_path)
    schedule: dict[tuple[int, int], bool] = {}
    for step, layers in payload["deltas_by_step"].items():
        step = int(step)
        if step == 0:
            continue
        sorted_layers = sorted(layers.items(), key=lambda kv: kv[1]["mean"])
        skip_layers = [int(l) for l, _ in sorted_layers[:k_per_step]]
        for l in skip_layers:
            schedule[(l, step)] = True
    return schedule
