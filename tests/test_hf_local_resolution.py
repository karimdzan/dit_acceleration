from pathlib import Path

from fae.utils.hf_local import repo_cache_dir, resolve_hf_file_path, resolve_hf_repo_root


def test_resolve_hf_repo_root_from_cache_main_ref(tmp_path: Path):
    repo_dir = repo_cache_dir('nyu-visionx/RAE-collections', cache_dir=tmp_path)
    snapshot = repo_dir / 'snapshots' / '123abc'
    snapshot.mkdir(parents=True)
    (repo_dir / 'refs').mkdir(parents=True, exist_ok=True)
    (repo_dir / 'refs' / 'main').write_text('123abc\n', encoding='utf-8')

    resolved = resolve_hf_repo_root('nyu-visionx/RAE-collections', cache_dir=tmp_path)
    assert resolved == snapshot


def test_resolve_hf_file_path_from_repo_root(tmp_path: Path):
    repo_root = tmp_path / 'RAE-collections'
    target = repo_root / 'decoders' / 'dinov2' / 'wReg_base' / 'ViTXL_n08' / 'model.pt'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b'123')

    resolved = resolve_hf_file_path(
        'nyu-visionx/RAE-collections',
        'decoders/dinov2/wReg_base/ViTXL_n08/model.pt',
        repo_root=repo_root,
    )
    assert resolved == target
