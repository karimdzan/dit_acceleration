"""Tests for the DiT-XL MLP TEAL-style activation sparsifier."""
import sys
import types

import torch
import torch.nn as nn


class StubFeedForward(nn.Module):
    def __init__(self, dim=8, hidden=32):
        super().__init__()
        self.up = nn.Linear(dim, hidden)
        self.act = nn.GELU(approximate="tanh")
        self.drop = nn.Identity()
        self.down = nn.Linear(hidden, dim)

        class UpThenAct(nn.Module):
            def __init__(self, up, act):
                super().__init__()
                self.up = up
                self.act = act

            def forward(self, x):
                return self.act(self.up(x))

        self.net = nn.ModuleList([
            UpThenAct(self.up, self.act),
            self.drop,
            self.down,
        ])

    def forward(self, x, *args, **kwargs):
        return self.net[2](self.net[1](self.net[0](x)))


class StubGEGLU(nn.Module):
    pass


def _setup_imports():
    diff = types.ModuleType("diffusers")
    models = types.ModuleType("diffusers.models")
    attn = types.ModuleType("diffusers.models.attention")
    acts = types.ModuleType("diffusers.models.activations")
    attn.FeedForward = StubFeedForward
    attn.GEGLU = StubGEGLU
    acts.GEGLU = StubGEGLU
    sys.modules.setdefault("diffusers", diff)
    sys.modules.setdefault("diffusers.models", models)
    sys.modules["diffusers.models.attention"] = attn
    sys.modules["diffusers.models.activations"] = acts


def _make_fake_pipe(n_blocks=3, dim=8, hidden=32):
    blocks = nn.ModuleList()
    for _ in range(n_blocks):
        b = nn.Module()
        b.ff = StubFeedForward(dim=dim, hidden=hidden)
        blocks.append(b)
    transformer = nn.Module()
    transformer.transformer_blocks = blocks
    transformer._dummy_param = nn.Parameter(torch.zeros(1))
    return types.SimpleNamespace(transformer=transformer)


def test_calibration_mode_passthrough_exactness():
    torch.manual_seed(0)
    pipe = _make_fake_pipe()
    x = torch.randn(2, 16, 8)
    ref = [b.ff(x.clone()) for b in pipe.transformer.transformer_blocks]

    from dit_accel.sparsity.dit_mlp_sparsifier import install_dit_mlp_sparsifier
    store = install_dit_mlp_sparsifier(pipe, thresholds=None, calibration_mode=True)
    got = [b.ff(x.clone()) for b in pipe.transformer.transformer_blocks]
    for r, g in zip(ref, got):
        assert torch.allclose(r, g, atol=0, rtol=0)
    assert len(store.calibration_samples) == 3


def test_inf_threshold_yields_bias_only():
    torch.manual_seed(1)
    pipe = _make_fake_pipe()
    bias = pipe.transformer.transformer_blocks[0].ff.down.bias.detach().clone()

    from dit_accel.sparsity.dit_mlp_sparsifier import install_dit_mlp_sparsifier
    thresholds = {
        f"transformer_blocks.{i}.ff": torch.tensor(float("inf"))
        for i in range(3)
    }
    install_dit_mlp_sparsifier(pipe, thresholds=thresholds)
    x = torch.randn(2, 16, 8)
    out = pipe.transformer.transformer_blocks[0].ff(x)
    expected = bias.view(1, 1, -1).expand_as(out)
    assert torch.allclose(out, expected, atol=1e-5)


def test_no_threshold_means_no_op():
    torch.manual_seed(2)
    pipe = _make_fake_pipe()
    x = torch.randn(2, 16, 8)
    ref = [b.ff(x.clone()) for b in pipe.transformer.transformer_blocks]

    from dit_accel.sparsity.dit_mlp_sparsifier import install_dit_mlp_sparsifier
    install_dit_mlp_sparsifier(pipe, thresholds=None, calibration_mode=False)
    got = [b.ff(x.clone()) for b in pipe.transformer.transformer_blocks]
    for r, g in zip(ref, got):
        assert torch.allclose(r, g, atol=0, rtol=0)


def test_fitted_threshold_hits_target_sparsity():
    torch.manual_seed(3)
    pipe = _make_fake_pipe()
    from dit_accel.sparsity.dit_mlp_sparsifier import install_dit_mlp_sparsifier
    from dit_accel.sparsity.store import fit_thresholds_from_samples

    store = install_dit_mlp_sparsifier(pipe, thresholds=None, calibration_mode=True)
    store.calibration_samples_per_batch = 4096
    for _ in range(4):
        x = torch.randn(2, 16, 8)
        for b in pipe.transformer.transformer_blocks:
            _ = b.ff(x)

    thresholds = fit_thresholds_from_samples(store.calibration_samples, target_sparsity=0.5)
    install_dit_mlp_sparsifier(pipe, thresholds=thresholds, calibration_mode=False)
    apply_store = pipe._dit_accel_sparsity
    for _ in range(4):
        x = torch.randn(2, 16, 8)
        for b in pipe.transformer.transformer_blocks:
            _ = b.ff(x)
    apply_store.flush()
    stats = apply_store.stats()
    assert 0.40 <= stats["global_sparsity"] <= 0.60


def main():
    _setup_imports()
    test_calibration_mode_passthrough_exactness()
    test_inf_threshold_yields_bias_only()
    test_no_threshold_means_no_op()
    test_fitted_threshold_hits_target_sparsity()
    print("OK")


if __name__ == "__main__":
    main()
