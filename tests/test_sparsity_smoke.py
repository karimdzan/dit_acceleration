"""Tests for the Mix-FFN activation sparsifier."""
import sys
import types

import torch
import torch.nn as nn


class StubGLUMBConv(nn.Module):
    def __init__(self, dim=8, hidden_channels=16, residual_connection=False, norm_type=None):
        super().__init__()
        self.dim = dim
        self.hidden_channels = hidden_channels
        self.residual_connection = residual_connection
        self.norm_type = norm_type
        self.conv_inverted = nn.Conv2d(dim, hidden_channels * 2, kernel_size=1)
        self.conv_depth = nn.Conv2d(
            hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
            groups=hidden_channels * 2,
        )
        self.conv_point = nn.Conv2d(hidden_channels, dim, kernel_size=1)
        self.nonlinearity = nn.SiLU()
        if norm_type == "rms_norm":
            self.norm = nn.LayerNorm(dim)

    def forward(self, hidden_states):
        residual = hidden_states if self.residual_connection else None
        hidden_states = self.conv_inverted(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv_depth(hidden_states)
        hidden_states, gate = torch.chunk(hidden_states, 2, dim=1)
        hidden_states = hidden_states * self.nonlinearity(gate)
        hidden_states = self.conv_point(hidden_states)
        if self.norm_type == "rms_norm":
            hidden_states = self.norm(hidden_states.movedim(1, -1)).movedim(-1, 1)
        if self.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states


def _setup_imports():
    pkg_root = types.ModuleType("diffusers")
    pkg_models = types.ModuleType("diffusers.models")
    pkg_xforms = types.ModuleType("diffusers.models.transformers")
    pkg_sana = types.ModuleType("diffusers.models.transformers.sana_transformer")
    pkg_sana.GLUMBConv = StubGLUMBConv
    sys.modules.setdefault("diffusers", pkg_root)
    sys.modules.setdefault("diffusers.models", pkg_models)
    sys.modules.setdefault("diffusers.models.transformers", pkg_xforms)
    sys.modules["diffusers.models.transformers.sana_transformer"] = pkg_sana


def _make_fake_transformer(n_blocks=3, dim=8, hidden_channels=16):
    blocks = nn.ModuleList()
    for _ in range(n_blocks):
        block = nn.Module()
        block.ff = StubGLUMBConv(dim=dim, hidden_channels=hidden_channels)
        blocks.append(block)
    transformer = nn.Module()
    transformer.transformer_blocks = blocks
    transformer._dummy_param = nn.Parameter(torch.zeros(1))
    pipe = types.SimpleNamespace(transformer=transformer)
    return pipe


def test_calibration_mode_passthrough_exactness():
    torch.manual_seed(0)
    pipe = _make_fake_transformer()
    x = torch.randn(2, 8, 4, 4)
    ref = [b.ff(x.clone()) for b in pipe.transformer.transformer_blocks]

    from dit_accel.sparsity.mixffn_sparsifier import install_mixffn_sparsifier
    store = install_mixffn_sparsifier(pipe, thresholds=None, calibration_mode=True)
    got = [b.ff(x.clone()) for b in pipe.transformer.transformer_blocks]

    for r, g in zip(ref, got):
        assert torch.allclose(r, g, atol=0, rtol=0)

    assert len(store.calibration_samples) == 3
    gate_numel = 2 * 16 * 4 * 4
    expected_k = min(store.calibration_samples_per_batch, gate_numel)
    for k, chunks in store.calibration_samples.items():
        assert len(chunks) == 1
        assert chunks[0].numel() == expected_k


def test_full_threshold_zeros_output_to_bias_only():
    torch.manual_seed(1)
    pipe = _make_fake_transformer()
    block0_conv_point = pipe.transformer.transformer_blocks[0].ff.conv_point
    bias = block0_conv_point.bias.detach().clone()

    from dit_accel.sparsity.mixffn_sparsifier import install_mixffn_sparsifier
    thresholds = {
        f"transformer_blocks.{i}.ff": torch.tensor(float("inf"))
        for i in range(3)
    }
    install_mixffn_sparsifier(pipe, thresholds=thresholds)
    x = torch.randn(2, 8, 4, 4)
    out = pipe.transformer.transformer_blocks[0].ff(x)

    expected = bias.view(1, -1, 1, 1).expand_as(out)
    assert torch.allclose(out, expected, atol=1e-5)


def test_no_threshold_means_no_op():
    torch.manual_seed(2)
    pipe = _make_fake_transformer()
    x = torch.randn(2, 8, 4, 4)
    ref = [b.ff(x.clone()) for b in pipe.transformer.transformer_blocks]

    from dit_accel.sparsity.mixffn_sparsifier import install_mixffn_sparsifier
    install_mixffn_sparsifier(pipe, thresholds=None, calibration_mode=False)
    got = [b.ff(x.clone()) for b in pipe.transformer.transformer_blocks]
    for r, g in zip(ref, got):
        assert torch.allclose(r, g, atol=0, rtol=0)


def test_fitted_threshold_hits_target_sparsity():
    torch.manual_seed(3)
    pipe = _make_fake_transformer()
    from dit_accel.sparsity.mixffn_sparsifier import install_mixffn_sparsifier
    from dit_accel.sparsity.store import fit_thresholds_from_samples

    store = install_mixffn_sparsifier(pipe, thresholds=None, calibration_mode=True)
    store.calibration_samples_per_batch = 4096
    for _ in range(4):
        x = torch.randn(2, 8, 4, 4)
        for b in pipe.transformer.transformer_blocks:
            _ = b.ff(x)

    thresholds = fit_thresholds_from_samples(
        store.calibration_samples, target_sparsity=0.5
    )

    install_mixffn_sparsifier(pipe, thresholds=thresholds, calibration_mode=False)
    apply_store = pipe._dit_accel_sparsity
    for _ in range(4):
        x = torch.randn(2, 8, 4, 4)
        for b in pipe.transformer.transformer_blocks:
            _ = b.ff(x)
    apply_store.flush()
    stats = apply_store.stats()
    assert 0.40 <= stats["global_sparsity"] <= 0.60
    for layer_id, rate in stats["per_layer_sparsity"].items():
        assert 0.30 <= rate <= 0.70


def main():
    _setup_imports()
    test_calibration_mode_passthrough_exactness()
    test_full_threshold_zeros_output_to_bias_only()
    test_no_threshold_means_no_op()
    test_fitted_threshold_hits_target_sparsity()
    print("OK")


if __name__ == "__main__":
    main()
