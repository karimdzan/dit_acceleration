import torch
from torch import nn

from dit_accel.sparsity.sana_sparse_ffn import (
    SanaFFNGroupSparseConfig,
    install_sana_ffn_group_observer,
    install_sana_ffn_group_sparse,
)


class DummyGLUMBConv(nn.Module):
    def __init__(self, dim=8, hidden=16):
        super().__init__()
        self.norm_type = None
        self.residual_connection = False
        self.nonlinearity = nn.SiLU()
        self.conv_inverted = nn.Conv2d(dim, hidden * 2, 1, 1, 0)
        self.conv_depth = nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1, groups=hidden * 2)
        self.conv_point = nn.Conv2d(hidden, dim, 1, 1, 0, bias=False)

    def forward(self, x):
        h = self.conv_inverted(x)
        h = self.nonlinearity(h)
        h = self.conv_depth(h)
        h, gate = torch.chunk(h, 2, dim=1)
        h = h * self.nonlinearity(gate)
        return self.conv_point(h)


class DummyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff = DummyGLUMBConv()


class DummyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([DummyBlock(), DummyBlock()])


class DummyPipe:
    def __init__(self):
        self.transformer = DummyTransformer()


def test_dynamic_keep_one_matches_dense():
    torch.manual_seed(0)
    pipe = DummyPipe()
    x = torch.randn(2, 8, 4, 4)
    dense = pipe.transformer.transformer_blocks[0].ff(x)
    install_sana_ffn_group_sparse(
        pipe,
        SanaFFNGroupSparseConfig(mode="dynamic", keep_ratio=1.0, group_size=4, verbose=False),
    )
    sparse = pipe.transformer.transformer_blocks[0].ff(x)
    assert torch.allclose(dense, sparse, atol=1e-5, rtol=1e-5)


def test_static_keep_all_matches_dense():
    torch.manual_seed(0)
    pipe = DummyPipe()
    x = torch.randn(2, 8, 4, 4)
    dense = pipe.transformer.transformer_blocks[1].ff(x)

    observer = install_sana_ffn_group_observer(pipe, group_size=4)
    _ = pipe.transformer.transformer_blocks[1].ff(x)
    plan = observer.build_plan(keep_ratio=1.0)

    # Install static from an in-memory plan by saving through torch's temp path.
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "plan.pt"
        torch.save(plan, path)
        install_sana_ffn_group_sparse(
            pipe,
            SanaFFNGroupSparseConfig(mode="static", keep_ratio=1.0, group_size=4, plan_path=path, verbose=False),
        )
        sparse = pipe.transformer.transformer_blocks[1].ff(x)

    assert torch.allclose(dense, sparse, atol=1e-5, rtol=1e-5)
