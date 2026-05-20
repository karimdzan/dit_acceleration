"""Shared state for activation-sparsity sparsifiers."""
from dataclasses import dataclass, field

import torch


@dataclass
class SparsityStore:
    thresholds: dict[str, torch.Tensor] = field(default_factory=dict)

    _dev_sparse: dict[str, torch.Tensor] = field(default_factory=dict)
    _dev_total: dict[str, torch.Tensor] = field(default_factory=dict)

    per_layer_sparse: dict[str, int] = field(default_factory=dict)
    per_layer_total: dict[str, int] = field(default_factory=dict)

    calibration_mode: bool = False
    use_kernel: bool = False
    calibration_samples: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    calibration_samples_per_batch: int = 4096

    def to(self, device) -> "SparsityStore":
        self.thresholds = {k: v.to(device) for k, v in self.thresholds.items()}
        return self

    def record(
        self,
        layer_id: str,
        n_sparse: torch.Tensor,
        n_total: int,
        device: torch.device,
    ) -> None:
        if layer_id not in self._dev_sparse:
            self._dev_sparse[layer_id] = torch.zeros((), dtype=torch.long, device=device)
            self._dev_total[layer_id] = torch.zeros((), dtype=torch.long, device=device)
        self._dev_sparse[layer_id] += n_sparse
        self._dev_total[layer_id] += n_total

    def record_calibration(
        self,
        layer_id: str,
        gate_magnitudes: torch.Tensor,
    ) -> None:
        flat = gate_magnitudes.detach().reshape(-1)
        n = flat.numel()
        k = min(self.calibration_samples_per_batch, n)
        idx = torch.randint(0, n, (k,), device=flat.device)
        sample = flat.index_select(0, idx).float().cpu()
        self.calibration_samples.setdefault(layer_id, []).append(sample)

    def flush(self) -> None:
        for k, v in self._dev_sparse.items():
            self.per_layer_sparse[k] = self.per_layer_sparse.get(k, 0) + int(v.item())
            v.zero_()
        for k, v in self._dev_total.items():
            self.per_layer_total[k] = self.per_layer_total.get(k, 0) + int(v.item())
            v.zero_()

    def stats(self) -> dict:
        total_sparse = sum(self.per_layer_sparse.values())
        total_elems = sum(self.per_layer_total.values())
        return {
            "global_sparsity": total_sparse / max(total_elems, 1),
            "total_zero_activations": total_sparse,
            "total_activations": total_elems,
            "n_layers": len(self.per_layer_total),
            "per_layer_sparsity": {
                k: self.per_layer_sparse.get(k, 0) / max(v, 1)
                for k, v in self.per_layer_total.items()
            },
        }


def fit_thresholds_from_samples(
    samples: dict[str, list[torch.Tensor]],
    target_sparsity: float,
) -> dict[str, torch.Tensor]:
    """Per-layer q-quantile thresholds at q = target_sparsity (CATS recipe)."""
    if not 0.0 <= target_sparsity < 1.0:
        raise ValueError(f"target_sparsity must be in [0, 1), got {target_sparsity}")

    thresholds: dict[str, torch.Tensor] = {}
    for layer_id, chunks in samples.items():
        all_vals = torch.cat(chunks)
        # torch.quantile has a 16M element limit.
        if all_vals.numel() > 10_000_000:
            idx = torch.randint(0, all_vals.numel(), (10_000_000,))
            all_vals = all_vals.index_select(0, idx)
        threshold = torch.quantile(all_vals, target_sparsity)
        thresholds[layer_id] = threshold
    return thresholds
