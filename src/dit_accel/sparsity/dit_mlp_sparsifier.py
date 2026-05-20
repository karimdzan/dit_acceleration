"""DiT-XL MLP activation sparsifier, TEAL-style."""
import torch

from .store import SparsityStore


_ORIG_FORWARD_ATTR = "_dit_accel_orig_forward"


def _make_sparse_ff_forward(orig_module, layer_id: str, store: SparsityStore):
    def forward(hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        m = orig_module
        h = m.net[0](hidden_states)
        h = m.net[1](h)

        if store.calibration_mode:
            store.record_calibration(layer_id, h.abs())
        else:
            threshold = store.thresholds.get(layer_id)
            if threshold is not None:
                mask = h.abs() >= threshold
                n_sparse = (~mask).sum()
                store.record(layer_id, n_sparse, mask.numel(), h.device)
                h = h * mask

        return m.net[2](h)

    return forward


def install_dit_mlp_sparsifier(
    pipe,
    thresholds: dict[str, torch.Tensor] | None = None,
    *,
    calibration_mode: bool = False,
) -> SparsityStore:
    from diffusers.models.attention import FeedForward
    try:
        from diffusers.models.activations import GEGLU
    except ImportError:
        from diffusers.models.attention import GEGLU

    store = SparsityStore()
    store.calibration_mode = calibration_mode
    if thresholds:
        store.thresholds = dict(thresholds)
        device = next(pipe.transformer.parameters()).device
        store.to(device)

    transformer = pipe.transformer
    blocks = getattr(transformer, "transformer_blocks", None)
    if blocks is None:
        raise RuntimeError("transformer.transformer_blocks not found.")

    installed = 0
    skipped_gated = 0
    for idx, block in enumerate(blocks):
        ff = getattr(block, "ff", None)
        if not isinstance(ff, FeedForward):
            continue
        if isinstance(ff.net[0], GEGLU):
            skipped_gated += 1
            continue
        layer_id = f"transformer_blocks.{idx}.ff"

        if hasattr(ff, _ORIG_FORWARD_ATTR):
            ff.forward = getattr(ff, _ORIG_FORWARD_ATTR)
            delattr(ff, _ORIG_FORWARD_ATTR)
        setattr(ff, _ORIG_FORWARD_ATTR, ff.forward)
        ff.forward = _make_sparse_ff_forward(ff, layer_id, store)
        installed += 1

    pipe._dit_accel_sparsity = store
    mode = "calibration" if calibration_mode else "apply"
    print(f"  installed DiT MLP sparsifier on {installed} layers (mode={mode})")
    if skipped_gated:
        print(f"  WARNING: skipped {skipped_gated} GEGLU blocks (need CATS-style sparsifier)")
    if not calibration_mode and not store.thresholds:
        print("  WARNING: apply mode but no thresholds loaded; sparsifier is a no-op")
    return store


def uninstall_dit_mlp_sparsifier(pipe) -> int:
    if not hasattr(pipe, "_dit_accel_sparsity"):
        return 0
    blocks = getattr(pipe.transformer, "transformer_blocks", [])
    n = 0
    for block in blocks:
        ff = getattr(block, "ff", None)
        if ff is not None and hasattr(ff, _ORIG_FORWARD_ATTR):
            ff.forward = getattr(ff, _ORIG_FORWARD_ATTR)
            delattr(ff, _ORIG_FORWARD_ATTR)
            n += 1
    delattr(pipe, "_dit_accel_sparsity")
    return n
