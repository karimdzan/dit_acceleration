import torch
import torch.nn as nn

from fae.modules.depth_pruning import apply_depth_pruning, select_layers_to_drop


class AddIndexBlock(nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def forward(self, x, *args, **kwargs):
        return x + self.idx


class DummyTransformer(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([AddIndexBlock(i) for i in range(n)])


def test_select_layers_uniform_preserves_edges():
    drops = select_layers_to_drop(10, {"strategy": "uniform", "drop_count": 3, "preserve_first": 1, "preserve_last": 1})
    assert len(drops) == 3
    assert 0 not in drops
    assert 9 not in drops


def test_apply_depth_pruning_removes_layers():
    model = DummyTransformer(8)
    result = apply_depth_pruning(model, {"enabled": True, "mode": "remove", "drop_layers": [2, 5]})
    assert result.original_num_layers == 8
    assert result.dropped_layers == [2, 5]
    assert result.kept_layers == [0, 1, 3, 4, 6, 7]
    assert len(model.transformer_blocks) == 6
    x = torch.tensor(0)
    for block in model.transformer_blocks:
        x = block(x)
    assert x.item() == sum([0, 1, 3, 4, 6, 7])


def test_apply_depth_pruning_skip_mode_keeps_length():
    model = DummyTransformer(5)
    result = apply_depth_pruning(model, {"enabled": True, "mode": "skip", "drop_layers": [1, 3]})
    assert len(model.transformer_blocks) == 5
    x = torch.tensor(0)
    for block in model.transformer_blocks:
        x = block(x)
    assert x.item() == 0 + 2 + 4
