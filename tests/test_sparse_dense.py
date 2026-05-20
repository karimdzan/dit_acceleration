import torch
import torch.nn as nn

from fae.modules.sparse_dense import (
    SparseDenseSanaBlockWrapper,
    apply_sparse_dense_token_blocks,
    downsample_tokens,
    upsample_tokens,
)


class RecordingSanaBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_shape = None
        self.last_hw = None

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
        height=None,
        width=None,
    ):
        self.last_shape = tuple(hidden_states.shape)
        self.last_hw = (height, width)
        return hidden_states + 1.0


class TinyTransformer(nn.Module):
    def __init__(self, n=4):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([RecordingSanaBlock() for _ in range(n)])


def test_downsample_upsample_round_trip_shape():
    x = torch.randn(2, 8 * 8, 4)
    sparse, hs, ws = downsample_tokens(x, height=8, width=8, stride=2)
    dense = upsample_tokens(sparse, sparse_height=hs, sparse_width=ws, height=8, width=8)
    assert sparse.shape == (2, 16, 4)
    assert (hs, ws) == (4, 4)
    assert dense.shape == x.shape


def test_sparse_dense_wrapper_runs_block_on_fewer_tokens_and_restores_shape():
    block = RecordingSanaBlock()
    wrapper = SparseDenseSanaBlockWrapper(block, layer_idx=0, stride=2, min_tokens=16)
    x = torch.randn(1, 8 * 8, 4)
    out = wrapper(x, height=8, width=8, timestep=torch.randn(1, 6, 4))
    assert out.shape == x.shape
    assert block.last_shape == (1, 16, 4)
    assert block.last_hw == (4, 4)
    assert wrapper.last_stats.original_tokens == 64
    assert wrapper.last_stats.sparse_tokens == 16


def test_sparse_dense_wrapper_falls_back_when_stride_invalid():
    block = RecordingSanaBlock()
    wrapper = SparseDenseSanaBlockWrapper(block, layer_idx=0, stride=3, min_tokens=16)
    x = torch.randn(1, 8 * 8, 4)
    out = wrapper(x, height=8, width=8, timestep=torch.randn(1, 6, 4))
    assert out.shape == x.shape
    assert block.last_shape == (1, 64, 4)
    assert block.last_hw == (8, 8)
    assert wrapper.last_stats is None


def test_apply_sparse_dense_token_blocks_wraps_selected_layers():
    model = TinyTransformer(5)
    wrapped = apply_sparse_dense_token_blocks(
        model,
        {"enabled": True, "layers": [1, 3], "stride": 2, "min_tokens": 16},
    )
    assert wrapped == [1, 3]
    assert isinstance(model.transformer_blocks[1], SparseDenseSanaBlockWrapper)
    assert isinstance(model.transformer_blocks[3], SparseDenseSanaBlockWrapper)
