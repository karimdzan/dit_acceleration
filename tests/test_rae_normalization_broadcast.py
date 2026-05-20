import torch

from fae.models.rae import RepresentationAutoEncoder


def test_channel_like_4d_stats_broadcast_over_grid():
    model = RepresentationAutoEncoder(
        input_dim=4,
        image_size=32,
        patch_size=8,
        normalize_latents=True,
        decoder_heads=1,
        decoder_layers=1,
        latent_mean=torch.zeros(1, 4, 1, 1),
        latent_var=torch.ones(1, 4, 1, 1),
    )
    x = torch.randn(2, 16, 4)
    z = model.normalize_latent_tokens(x)
    assert z.shape == x.shape


def test_spatial_stats_must_be_broadcastable():
    model = RepresentationAutoEncoder(input_dim=4, image_size=32, patch_size=8, normalize_latents=True, decoder_heads=1, decoder_layers=1)
    model.set_latent_stats(torch.zeros(4, 2, 3), torch.ones(4, 2, 3))
    x = torch.randn(2, 16, 4)
    try:
        model.normalize_latent_tokens(x)
    except ValueError as exc:
        assert 'broadcastable' in str(exc)
    else:
        raise AssertionError('Expected a broadcastability error')
