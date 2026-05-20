"""Sana Mix-FFN (GLUMBConv) activation sparsifier, CATS-style."""
import torch

from .store import SparsityStore


_ORIG_FORWARD_ATTR = "_dit_accel_orig_forward"


def _make_sparse_forward(orig_module, layer_id: str, store: SparsityStore):
    def forward(hidden_states: torch.Tensor) -> torch.Tensor:
        m = orig_module
        residual = hidden_states if m.residual_connection else None

        hidden_states = m.conv_inverted(hidden_states)
        hidden_states = m.nonlinearity(hidden_states)
        hidden_states = m.conv_depth(hidden_states)
        value, gate = torch.chunk(hidden_states, 2, dim=1)
        gate_act = m.nonlinearity(gate)

        mask = None
        if store.calibration_mode:
            store.record_calibration(layer_id, gate_act.abs())
            hidden_states = value * gate_act
        else:
            threshold = store.thresholds.get(layer_id)
            if threshold is not None:
                mask = gate_act.abs() >= threshold
                n_sparse = (~mask).sum()
                store.record(layer_id, n_sparse, mask.numel(), gate_act.device)
                hidden_states = (value * gate_act) * mask
            else:
                hidden_states = value * gate_act

        if store.use_kernel and mask is not None:
            from .triton_kernel import sparse_conv_point
            hidden_states = sparse_conv_point(
                hidden_states,
                m.conv_point.weight,
                m.conv_point.bias,
                spatial_mask=mask,
            )
        else:
            hidden_states = m.conv_point(hidden_states)

        if m.norm_type == "rms_norm":
            hidden_states = m.norm(hidden_states.movedim(1, -1)).movedim(-1, 1)
        if m.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states

    return forward


def install_mixffn_sparsifier(
    pipe,
    thresholds: dict[str, torch.Tensor] | None = None,
    *,
    calibration_mode: bool = False,
    use_kernel: bool = False,
) -> SparsityStore:
    from diffusers.models.transformers.sana_transformer import GLUMBConv

    store = SparsityStore()
    store.calibration_mode = calibration_mode
    store.use_kernel = use_kernel
    if thresholds:
        store.thresholds = dict(thresholds)
        device = next(pipe.transformer.parameters()).device
        store.to(device)

    transformer = pipe.transformer
    installed = 0
    for idx, block in enumerate(transformer.transformer_blocks):
        ff = block.ff
        if not isinstance(ff, GLUMBConv):
            continue
        layer_id = f"transformer_blocks.{idx}.ff"

        if hasattr(ff, _ORIG_FORWARD_ATTR):
            ff.forward = getattr(ff, _ORIG_FORWARD_ATTR)
            delattr(ff, _ORIG_FORWARD_ATTR)

        setattr(ff, _ORIG_FORWARD_ATTR, ff.forward)
        ff.forward = _make_sparse_forward(ff, layer_id, store)
        installed += 1

    pipe._dit_accel_sparsity = store
    mode = "calibration" if calibration_mode else "apply"
    print(f"  installed Mix-FFN sparsifier on {installed} layers (mode={mode})")
    if not calibration_mode and not store.thresholds:
        print("  WARNING: apply mode but no thresholds loaded; sparsifier is a no-op")
    return store


def uninstall_mixffn_sparsifier(pipe) -> int:
    if not hasattr(pipe, "_dit_accel_sparsity"):
        return 0
    n = 0
    for block in pipe.transformer.transformer_blocks:
        ff = block.ff
        if hasattr(ff, _ORIG_FORWARD_ATTR):
            ff.forward = getattr(ff, _ORIG_FORWARD_ATTR)
            delattr(ff, _ORIG_FORWARD_ATTR)
            n += 1
    delattr(pipe, "_dit_accel_sparsity")
    return n
