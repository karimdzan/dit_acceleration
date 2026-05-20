import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from fae.generators.common import ConditioningBundle, LatentTensorSpec
from fae.generators.internal import InternalDiTDHBackend


def test_internal_ditdh_forward_and_loss():
    spec = LatentTensorSpec(channels=768, height=16, width=16)
    backend = InternalDiTDHBackend(
        spec=spec,
        hidden_size=(256, 384),
        depth=(2, 1),
        num_heads=(8, 8),
        num_classes=1000,
        class_dropout_prob=0.1,
        objective="linear_velocity",
    )
    x = torch.randn(2, 768, 16, 16)
    cond = ConditioningBundle(class_labels=torch.tensor([1, 2], dtype=torch.long))
    loss_out = backend.training_loss(x, conditioning=cond)
    assert torch.is_tensor(loss_out.loss)
    assert loss_out.loss.ndim == 0
    with torch.no_grad():
        y = backend.model(x, torch.rand(2), cond)
    assert y.shape == x.shape



def test_internal_ditdh_zero_initialized_output_and_modulation():
    spec = LatentTensorSpec(channels=32, height=4, width=4)
    backend = InternalDiTDHBackend(
        spec=spec,
        hidden_size=(64, 96),
        depth=(2, 1),
        num_heads=(8, 8),
        num_classes=10,
        class_dropout_prob=0.0,
        objective="linear_velocity",
    )
    model = backend.model
    assert torch.count_nonzero(model.final_layer.linear.weight) == 0
    assert torch.count_nonzero(model.final_layer.linear.bias) == 0
    for block in list(model.encoder_blocks) + list(model.decoder_blocks):
        assert torch.count_nonzero(block.adaLN_modulation[-1].weight) == 0
        assert torch.count_nonzero(block.adaLN_modulation[-1].bias) == 0

    x = torch.randn(2, 32, 4, 4)
    cond = ConditioningBundle(class_labels=torch.tensor([1, 2], dtype=torch.long))
    with torch.no_grad():
        y = model(x, torch.rand(2), cond)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)
