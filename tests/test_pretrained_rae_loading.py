from pathlib import Path

import torch

from fae.models.rae import RepresentationAutoEncoder
from fae.utils.pretrained import initialize_rae_from_pretrained, resolve_rae_assets


def _make_model() -> RepresentationAutoEncoder:
    return RepresentationAutoEncoder(
        input_dim=8,
        image_size=32,
        patch_size=16,
        decoder_hidden_dim=16,
        decoder_layers=1,
        decoder_heads=2,
        decoder_use_rope_2d=False,
        normalize_latents=True,
    )


def test_decoder_only_pretrained_loads_without_external_stats_file(tmp_path: Path):
    model = _make_model()
    decoder_ckpt = tmp_path / 'decoder.pt'

    state = {'decoder.' + k: v.clone() for k, v in model.decoder.state_dict().items()}
    payload = {
        'model': state,
        'latent_stats': {
            'mean': torch.zeros(8, 2, 2),
            'var': torch.ones(8, 2, 2),
        },
    }
    torch.save(payload, decoder_ckpt)

    config = {
        'fae': {
            'pretrained': {
                'decoder_path': str(decoder_ckpt),
                'strict': True,
            },
            'normalize_latents': True,
        },
        'pixel_decoder': {
            'image_size': 32,
            'patch_size': 16,
            'hidden_dim': 16,
            'num_layers': 1,
            'num_heads': 2,
            'use_rope_2d': False,
        },
    }
    assets = resolve_rae_assets(config)
    assert assets.decoder_filename is None
    assert assets.decoder_path == str(decoder_ckpt)

    new_model = _make_model()
    info = initialize_rae_from_pretrained(new_model, config, strict=True)
    assert info['loaded'] is True
    assert info['mode'] == 'decoder'
    assert info['latent_stats_loaded'] is True
    assert tuple(new_model.latent_mean.shape[-2:]) == (2, 2)
    for key, value in model.decoder.state_dict().items():
        assert torch.equal(value, new_model.decoder.state_dict()[key])


def test_preset_resolves_exact_hf_relative_filenames_from_snapshot_dir(tmp_path: Path):
    snapshot = tmp_path / 'snapshots' / 'abcdef'
    decoder_file = snapshot / 'decoders' / 'dinov2' / 'wReg_base' / 'ViTXL_n08' / 'model.pt'
    generator_file = snapshot / 'DiTs' / 'Dinov2' / 'wReg_base' / 'ImageNet256' / 'DiTDH-XL' / 'stage2_model.pt'
    decoder_file.parent.mkdir(parents=True, exist_ok=True)
    generator_file.parent.mkdir(parents=True, exist_ok=True)
    decoder_file.write_bytes(b'decoder')
    generator_file.write_bytes(b'generator')

    config = {
        'fae': {
            'pretrained': {
                'preset': 'dinov2_wreg_base_imagenet256_vitxl_n08',
                'repo_id': 'nyu-visionx/RAE-collections',
                'snapshot_dir': str(snapshot),
            }
        },
    }
    assets = resolve_rae_assets(config)
    assert assets.decoder_filename == 'decoders/dinov2/wReg_base/ViTXL_n08/model.pt'
    assert assets.generator_filename == 'DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-XL/stage2_model.pt'
    assert assets.decoder_path == str(decoder_file)
    assert assets.generator_path == str(generator_file)


def test_local_stage3_checkpoint_does_not_override_local_checkpoint(tmp_path: Path):
    missing_local = tmp_path / 'missing_local_latest.pt'
    snapshot = tmp_path / 'snapshot'
    decoder_file = snapshot / 'decoders' / 'dinov2' / 'wReg_base' / 'ViTXL_n08' / 'model.pt'
    generator_file = snapshot / 'DiTs' / 'Dinov2' / 'wReg_base' / 'ImageNet256' / 'DiTDH-XL' / 'stage2_model.pt'
    decoder_file.parent.mkdir(parents=True, exist_ok=True)
    generator_file.parent.mkdir(parents=True, exist_ok=True)
    decoder_file.write_bytes(b'decoder')
    generator_file.write_bytes(b'generator')
    config = {
        'fae': {
            'pretrained': {
                'preset': 'dinov2_wreg_base_imagenet256_vitxl_n08',
                'repo_id': 'nyu-visionx/RAE-collections',
                'snapshot_dir': str(snapshot),
            }
        },
        'stage3': {
            'autoencoder_checkpoint': str(missing_local),
        },
    }
    assets = resolve_rae_assets(config)
    assert assets.autoencoder_checkpoint == str(missing_local)
