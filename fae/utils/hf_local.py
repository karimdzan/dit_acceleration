import os
from pathlib import Path
from typing import Any


class HFLocalResolutionError(RuntimeError):
    pass


def _expand(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    return Path(path).expanduser()


def default_hf_home() -> Path:
    env = os.environ.get("HF_HOME")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "huggingface"
    return Path.home() / ".cache" / "huggingface"


def default_hf_hub_cache() -> Path:
    env = os.environ.get("HF_HUB_CACHE")
    if env:
        return Path(env).expanduser()
    return default_hf_home() / "hub"


def repo_cache_dir(repo_id: str, *, cache_dir: str | Path | None = None, repo_type: str = "model") -> Path:
    prefix_map = {
        "model": "models",
        "dataset": "datasets",
        "space": "spaces",
    }
    if repo_type not in prefix_map:
        raise ValueError(f"Unsupported repo_type={repo_type!r}")
    root = _expand(cache_dir) or default_hf_hub_cache()
    safe_repo = repo_id.replace("/", "--")
    return root / f"{prefix_map[repo_type]}--{safe_repo}"


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _resolve_snapshot_dir_from_cache(
    repo_id: str,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    repo_type: str = "model",
) -> Path | None:
    repo_dir = repo_cache_dir(repo_id, cache_dir=cache_dir, repo_type=repo_type)
    snapshots_dir = repo_dir / "snapshots"
    refs_dir = repo_dir / "refs"
    if not snapshots_dir.exists():
        return None

    if revision:
        direct = snapshots_dir / revision
        if direct.exists():
            return direct
        ref_target = _read_text(refs_dir / revision)
        if ref_target:
            resolved = snapshots_dir / ref_target
            if resolved.exists():
                return resolved
        return None

    main_ref = _read_text(refs_dir / "main")
    if main_ref:
        resolved = snapshots_dir / main_ref
        if resolved.exists():
            return resolved

    candidates = sorted((p for p in snapshots_dir.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise HFLocalResolutionError(
            f"Multiple cached snapshots found for {repo_id} under {snapshots_dir}. "
            "Set pretrained.revision or pretrained.snapshot_dir to disambiguate."
        )
    return None


def resolve_hf_repo_root(
    repo_id: str,
    *,
    snapshot_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    repo_type: str = "model",
    require_local: bool = True,
) -> Path | None:
    explicit_repo_root = _expand(repo_root)
    if explicit_repo_root is not None:
        if explicit_repo_root.exists():
            return explicit_repo_root
        if require_local:
            raise HFLocalResolutionError(f"Configured repo_root does not exist: {explicit_repo_root}")
        return None

    explicit_snapshot = _expand(snapshot_dir)
    if explicit_snapshot is not None:
        if explicit_snapshot.exists():
            return explicit_snapshot
        if require_local:
            raise HFLocalResolutionError(f"Configured snapshot_dir does not exist: {explicit_snapshot}")
        return None

    resolved = _resolve_snapshot_dir_from_cache(
        repo_id,
        revision=revision,
        cache_dir=cache_dir,
        repo_type=repo_type,
    )
    if resolved is not None:
        return resolved

    if require_local:
        repo_dir = repo_cache_dir(repo_id, cache_dir=cache_dir, repo_type=repo_type)
        raise HFLocalResolutionError(
            f"Could not resolve a local Hugging Face snapshot for {repo_id}. "
            f"Looked under {repo_dir}. Download the repo first with `hf download`, or set snapshot_dir/repo_root explicitly."
        )
    return None


def resolve_hf_file_path(
    repo_id: str,
    filename: str | Path,
    *,
    snapshot_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    repo_type: str = "model",
    require_local: bool = True,
) -> Path | None:
    file_path = _expand(filename)
    if file_path is None:
        return None
    if file_path.is_absolute() or file_path.exists():
        if file_path.exists():
            return file_path
        if require_local:
            raise HFLocalResolutionError(f"Configured path does not exist: {file_path}")
        return None

    explicit_root = _expand(repo_root)
    if explicit_root is not None:
        candidate = explicit_root / file_path
        if candidate.exists():
            return candidate
        if require_local:
            raise HFLocalResolutionError(f"Could not find {file_path} under repo_root={explicit_root}")
        return None

    explicit_snapshot = _expand(snapshot_dir)
    if explicit_snapshot is not None:
        candidate = explicit_snapshot / file_path
        if candidate.exists():
            return candidate
        if require_local:
            raise HFLocalResolutionError(f"Could not find {file_path} under snapshot_dir={explicit_snapshot}")
        return None

    try:
        from huggingface_hub import _CACHED_NO_EXIST, try_to_load_from_cache  # type: ignore
    except Exception:
        cached = None
    else:
        cached = try_to_load_from_cache(
            repo_id=repo_id,
            filename=str(file_path).replace("\\", "/"),
            revision=revision,
            repo_type=repo_type,
            cache_dir=str(_expand(cache_dir)) if cache_dir is not None else None,
        )
        if isinstance(cached, str):
            return Path(cached)
        if cached is _CACHED_NO_EXIST:
            if require_local:
                raise HFLocalResolutionError(f"{file_path} is cached as missing for repo {repo_id}.")
            return None

    root = resolve_hf_repo_root(
        repo_id,
        snapshot_dir=snapshot_dir,
        repo_root=repo_root,
        revision=revision,
        cache_dir=cache_dir,
        repo_type=repo_type,
        require_local=require_local,
    )
    if root is None:
        return None
    candidate = root / file_path
    if candidate.exists():
        return candidate
    if require_local:
        raise HFLocalResolutionError(
            f"Could not find {file_path} inside the resolved local Hugging Face repo root {root}."
        )
    return None


def resolve_from_hf_config(
    cfg: dict[str, Any],
    *,
    repo_id_key: str = "repo_id",
    revision_key: str = "revision",
    cache_dir_key: str = "cache_dir",
    snapshot_dir_key: str = "snapshot_dir",
    repo_root_key: str = "repo_root",
    local_dir_alias: str = "local_dir",
    require_local: bool = True,
) -> tuple[str | None, Path | None]:
    repo_id = cfg.get(repo_id_key)
    if repo_id is None:
        return None, None
    repo_root = cfg.get(repo_root_key) or cfg.get(local_dir_alias)
    root = resolve_hf_repo_root(
        str(repo_id),
        snapshot_dir=cfg.get(snapshot_dir_key),
        repo_root=repo_root,
        revision=cfg.get(revision_key),
        cache_dir=cfg.get(cache_dir_key),
        require_local=require_local,
    )
    return str(repo_id), root
