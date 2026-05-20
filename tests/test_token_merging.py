from pathlib import Path
import sys

repo_root = Path("/home/kaytkhadzhayev/fae/")
repo_root = repo_root.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from fae.modules.token_merging import (
    SpatialTokenMergingWrapper,
    SpatialTokenPooler,
    apply_spatial_token_merging,
    iter_token_merging_wrappers,
)



class RecordingBlock(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.seen_tokens = None
        self.seen_grid_size = None

    def forward(self, x, cond=None, grid_size=None):
        self.seen_tokens = x.shape[1]
        self.seen_grid_size = grid_size
        return self.proj(x)


class HiddenStatesRecordingBlock(RecordingBlock):
    def forward(self, hidden_states, cond=None, grid_size=None):
        return super().forward(hidden_states, cond=cond, grid_size=grid_size)


class DummyDiT(nn.Module):
    def __init__(self, depth=4, dim=8):
        super().__init__()
        self.blocks = nn.ModuleList([RecordingBlock(dim=dim) for _ in range(depth)])

    def forward(self, x, cond=None, grid_size=(8, 8)):
        for block in self.blocks:
            x = block(x, cond, grid_size=grid_size)
        return x


def test_spatial_pooler_preserves_shape_after_unmerge():
    pooler = SpatialTokenPooler(stride=2)
    x = torch.randn(2, 8 * 8, 16)
    merged, stats = pooler.merge(x, grid_size=(8, 8))
    restored = pooler.unmerge(merged, stats)

    assert merged.shape == (2, 4 * 4, 16)
    assert restored.shape == x.shape
    assert stats.original_tokens == 64
    assert stats.merged_tokens == 16
    assert stats.keep_ratio == 0.25


def test_wrapper_reduces_tokens_inside_block_and_restores_output_shape():
    block = RecordingBlock(dim=8)
    wrapped = SpatialTokenMergingWrapper(block, stride=2)
    x = torch.randn(2, 64, 8)
    out = wrapped(x, None, grid_size=(8, 8))

    assert out.shape == x.shape
    assert block.seen_tokens == 16
    assert block.seen_grid_size == (4, 4)
    assert wrapped.last_stats is not None
    assert wrapped.last_stats.original_tokens == 64
    assert wrapped.last_stats.merged_tokens == 16


def test_wrapper_supports_diffusers_hidden_states_keyword_style():
    block = HiddenStatesRecordingBlock(dim=8)
    wrapped = SpatialTokenMergingWrapper(block, stride=2)
    x = torch.randn(2, 64, 8)
    out = wrapped(hidden_states=x, cond=None, grid_size=(8, 8))

    assert out.shape == x.shape
    assert block.seen_tokens == 16


def test_apply_spatial_token_merging_wraps_selected_layers_only():
    model = DummyDiT(depth=6, dim=8)
    wrapped_layers = apply_spatial_token_merging(
        model,
        {
            "enabled": True,
            "method": "spatial_pool",
            "stride": 2,
            "start_layer": 1,
            "end_layer": 5,
            "every": 2,
        },
    )

    assert wrapped_layers == [1, 3]
    assert isinstance(model.blocks[1], SpatialTokenMergingWrapper)
    assert isinstance(model.blocks[3], SpatialTokenMergingWrapper)
    assert not isinstance(model.blocks[0], SpatialTokenMergingWrapper)
    assert not isinstance(model.blocks[5], SpatialTokenMergingWrapper)



def test_dummy_dit_forward_records_reduced_tokens_on_wrapped_layers():
    model = DummyDiT(depth=4, dim=8)
    apply_spatial_token_merging(
        model,
        {
            "enabled": True,
            "method": "spatial_pool",
            "stride": 2,
            "start_layer": 1,
            "end_layer": 3,
        },
    )
    x = torch.randn(2, 64, 8)
    out = model(x, grid_size=(8, 8))

    assert out.shape == x.shape
    assert model.blocks[0].seen_tokens == 64
    assert model.blocks[1].block.seen_tokens == 16
    assert model.blocks[2].block.seen_tokens == 16
    assert model.blocks[3].seen_tokens == 64
    wrappers = list(iter_token_merging_wrappers(model))
    assert len(wrappers) == 2
    assert all(w.last_stats is not None for w in wrappers)
    assert all(w.last_stats.merged_grid_size == (4, 4) for w in wrappers)
