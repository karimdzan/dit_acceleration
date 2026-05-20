import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from fae.models.rae import RepresentationAutoEncoder


def test_spatial_latent_normalization_roundtrip():
    mean = torch.randn(8, 4, 4)
    var = torch.rand(8, 4, 4).abs() + 0.5
    model = RepresentationAutoEncoder(
        input_dim=8,
        image_size=64,
        patch_size=16,
        decoder_hidden_dim=16,
        decoder_layers=1,
        decoder_heads=4,
        normalize_latents=True,
        latent_mean=mean,
        latent_var=var,
    )
    z = torch.randn(2, 16, 8)
    z_norm = model.normalize_latent_tokens(z)
    z_back = model.denormalize_latent_tokens(z_norm)
    assert torch.allclose(z, z_back, atol=1e-5, rtol=1e-5)
