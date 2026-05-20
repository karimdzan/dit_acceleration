from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from fae.utils.checkpoint import load_checkpoint
from fae.utils.hf_local import HFLocalResolutionError, resolve_hf_file_path


OFFICIAL_RAE_PRESETS: dict[str, dict[str, str]] = {
    "dinov2_wreg_base_imagenet256_vitxl_n08": {
        "repo_id": "nyu-visionx/RAE-collections",
        "decoder_path": "decoders/dinov2/wReg_base/ViTXL_n08/model.pt",
        "generator_path": "DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-XL/stage2_model.pt",
    },
    "dinov2_wreg_base_imagenet512_vitxl_n08": {
        "repo_id": "nyu-visionx/RAE-collections",
        "decoder_path": "decoders/dinov2/wReg_base/ViTXL_n08_i512/model.pt",
    },
    "siglip2_base_i256_vitxl_n08": {
        "repo_id": "nyu-visionx/RAE-collections",
        "decoder_path": "decoders/siglip2/base_p16_i256/ViTXL_n08/model.pt",
        "generator_path": "DiTs/SigLIP2/b16/ImageNet256/DiT-XL/stage2_model.pt",
    },
    "mae_base_p16_vitxl_n08": {
        "repo_id": "nyu-visionx/RAE-collections",
        "decoder_path": "decoders/mae/base_p16/ViTXL_n08/model.pt",
        "generator_path": "DiTs/MAE/b16/ImageNet256/DiT-XL/stage2_model.pt",
    },
}


@dataclass
class ResolvedRAEAssets:
    decoder_path: str | None = None
    decoder_filename: str | None = None
    autoencoder_checkpoint: str | None = None
    generator_path: str | None = None
    generator_filename: str | None = None
    source: str = "none"
    repo_id: str | None = None
    revision: str | None = None
    repo_root: str | None = None


class PretrainedAssetResolutionError(RuntimeError):
    pass


class PretrainedAssetLoadError(RuntimeError):
    pass


def _as_path_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    text = str(value).strip()
    return text or None


def _get_ae_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("rae") or config.get("fae") or {})


def _get_pretrained_cfg(config: dict[str, Any]) -> dict[str, Any]:
    ae_cfg = _get_ae_cfg(config)
    raw = ae_cfg.get("pretrained")
    if isinstance(raw, dict):
        return dict(raw)
    if raw:
        return {"preset": raw}
    out: dict[str, Any] = {}
    for src_key, dst_key in (
        ("pretrained_decoder_path", "decoder_path"),
        ("pretrained_autoencoder_checkpoint", "autoencoder_checkpoint"),
        ("pretrained_generator_path", "generator_path"),
        ("pretrained_repo_id", "repo_id"),
        ("pretrained_revision", "revision"),
        ("pretrained_cache_dir", "cache_dir"),
        ("pretrained_local_dir", "repo_root"),
        ("pretrained_snapshot_dir", "snapshot_dir"),
        ("pretrained_preset", "preset"),
        ("pretrained_strict", "strict"),
        ("pretrained_load_ema", "load_ema"),
    ):
        if src_key in ae_cfg:
            out[dst_key] = ae_cfg[src_key]
    return out


def _resolve_hf_relative_path(
    repo_id: str | None,
    filename: str | None,
    *,
    revision: str | None,
    cache_dir: str | None,
    snapshot_dir: str | None,
    repo_root: str | None,
) -> tuple[str | None, str | None]:
    if not filename:
        return None, None
    if Path(filename).exists() or Path(filename).is_absolute():
        path = Path(filename)
        if path.exists():
            return str(path), None
        raise PretrainedAssetResolutionError(f"Configured path does not exist: {path}")
    if not repo_id:
        return filename, filename
    try:
        resolved = resolve_hf_file_path(
            repo_id,
            filename,
            revision=revision,
            cache_dir=cache_dir,
            snapshot_dir=snapshot_dir,
            repo_root=repo_root,
            require_local=True,
        )
    except HFLocalResolutionError as exc:
        details = str(exc)
        search_roots = [x for x in (snapshot_dir, repo_root) if x]
        suggestions: list[str] = []
        wanted_name = Path(filename).name
        wanted_parent = str(Path(filename).parent)
        for root_dir in search_roots:
            base = Path(root_dir)
            if not base.exists():
                continue
            try:
                if wanted_parent and (base / wanted_parent).exists():
                    suggestions.extend(sorted(str(path.relative_to(base)) for path in (base / wanted_parent).glob('*') if path.is_file())[:10])
                else:
                    suggestions.extend(sorted(str(path.relative_to(base)) for path in base.rglob(wanted_name))[:10])
            except Exception:
                pass
        if suggestions:
            details += f". Nearby local files: {suggestions}"
        raise PretrainedAssetResolutionError(details) from exc
    return str(resolved), filename


def resolve_rae_assets(config: dict[str, Any]) -> ResolvedRAEAssets:
    ae_cfg = _get_ae_cfg(config)
    pt_cfg = _get_pretrained_cfg(config)

    preset_name = _as_path_str(pt_cfg.get("preset"))
    preset = OFFICIAL_RAE_PRESETS.get(preset_name, {}) if preset_name else {}

    stage2_cfg = dict(config.get("stage2") or {})
    stage3_cfg = dict(config.get("stage3") or {})

    repo_id = _as_path_str(pt_cfg.get("repo_id")) or preset.get("repo_id")
    revision = _as_path_str(pt_cfg.get("revision"))
    cache_dir = _as_path_str(pt_cfg.get("cache_dir"))
    snapshot_dir = _as_path_str(pt_cfg.get("snapshot_dir"))
    repo_root = _as_path_str(pt_cfg.get("repo_root") or pt_cfg.get("local_dir"))

    decoder_value, decoder_origin = None, None
    for origin, value in (
        ("pretrained", _as_path_str(pt_cfg.get("decoder_path") or pt_cfg.get("decoder_filename"))),
        ("pretrained", _as_path_str(ae_cfg.get("pretrained_decoder_path"))),
        ("local", _as_path_str(ae_cfg.get("decoder_checkpoint"))),
        ("preset", preset.get("decoder_path")),
    ):
        if value:
            decoder_value, decoder_origin = value, origin
            break

    autoencoder_checkpoint, autoencoder_origin = None, None
    for origin, value in (
        ("pretrained", _as_path_str(pt_cfg.get("autoencoder_checkpoint"))),
        ("pretrained", _as_path_str(ae_cfg.get("pretrained_autoencoder_checkpoint"))),
        ("local", _as_path_str(stage3_cfg.get("autoencoder_checkpoint"))),
        ("local", _as_path_str(stage3_cfg.get("fae_checkpoint"))),
        ("local", _as_path_str(stage2_cfg.get("autoencoder_checkpoint"))),
        ("local", _as_path_str(stage2_cfg.get("fae_checkpoint"))),
        ("local", _as_path_str(config.get("stage1", {}).get("checkpoint"))),
    ):
        if value:
            autoencoder_checkpoint, autoencoder_origin = value, origin
            break

    generator_value, generator_origin = None, None
    for origin, value in (
        ("pretrained", _as_path_str(pt_cfg.get("generator_path") or pt_cfg.get("generator_filename"))),
        ("pretrained", _as_path_str(ae_cfg.get("pretrained_generator_path"))),
        ("local", _as_path_str(stage3_cfg.get("generator_checkpoint"))),
        ("preset", preset.get("generator_path")),
    ):
        if value:
            generator_value, generator_origin = value, origin
            break

    source = "local"
    decoder_filename = None
    decoder_path = decoder_value
    if decoder_value and decoder_origin in {"preset", "pretrained"}:
        decoder_path, decoder_filename = _resolve_hf_relative_path(
            repo_id,
            decoder_value,
            revision=revision,
            cache_dir=cache_dir,
            snapshot_dir=snapshot_dir,
            repo_root=repo_root,
        )
        source = "hf_local"

    generator_filename = None
    generator_path = generator_value
    if generator_value and generator_origin in {"preset", "pretrained"}:
        generator_path, generator_filename = _resolve_hf_relative_path(
            repo_id,
            generator_value,
            revision=revision,
            cache_dir=cache_dir,
            snapshot_dir=snapshot_dir,
            repo_root=repo_root,
        )
        source = "hf_local"

    if autoencoder_checkpoint and autoencoder_origin in {"preset", "pretrained"}:
        autoencoder_checkpoint, _ = _resolve_hf_relative_path(
            repo_id,
            autoencoder_checkpoint,
            revision=revision,
            cache_dir=cache_dir,
            snapshot_dir=snapshot_dir,
            repo_root=repo_root,
        )
        source = "hf_local"

    if source == "local" and preset_name:
        source = "preset"

    return ResolvedRAEAssets(
        decoder_path=decoder_path,
        decoder_filename=decoder_filename,
        autoencoder_checkpoint=autoencoder_checkpoint,
        generator_path=generator_path,
        generator_filename=generator_filename,
        source=source,
        repo_id=repo_id,
        revision=revision,
        repo_root=repo_root or snapshot_dir,
    )


_STATE_KEYS = (
    "ema_model",
    "model",
    "state_dict",
    "module",
    "autoencoder",
    "decoder",
)


def unwrap_state_dict(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise PretrainedAssetLoadError(f"Expected checkpoint-like dict, got {type(obj)!r}.")
    if obj and all(isinstance(k, str) for k in obj.keys()) and any(torch.is_tensor(v) for v in obj.values()):
        return obj
    for key in _STATE_KEYS:
        value = obj.get(key)
        if isinstance(value, dict):
            return unwrap_state_dict(value)
    raise PretrainedAssetLoadError(
        f"Could not find a tensor state dict in checkpoint. Available keys: {sorted(obj.keys())}"
    )


def _strip_prefix_if_present(state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    plen = len(prefix)
    return {k[plen:]: v for k, v in state_dict.items() if k.startswith(prefix)}


def _candidate_state_dicts_for_module(state_dict: dict[str, Any], prefixes: list[str]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for prefix in prefixes:
        if prefix == "":
            candidates.append(state_dict)
            continue
        stripped = _strip_prefix_if_present(state_dict, prefix)
        if stripped:
            candidates.append(stripped)
    return candidates


def _best_state_dict_for_module(module: torch.nn.Module, state_dict: dict[str, Any], prefixes: list[str]) -> tuple[dict[str, Any], dict[str, list[str]]]:
    target_keys = set(module.state_dict().keys())
    best_sd = state_dict
    best_score = (-1, 10**9)
    best_info = {"missing": list(target_keys), "unexpected": list(state_dict.keys())}
    for candidate in _candidate_state_dicts_for_module(state_dict, prefixes):
        candidate_keys = set(candidate.keys())
        overlap = len(target_keys & candidate_keys)
        unexpected = len(candidate_keys - target_keys)
        missing = len(target_keys - candidate_keys)
        score = (overlap, -(missing + unexpected))
        if score > best_score:
            best_score = score
            best_sd = candidate
            best_info = {
                "missing": sorted(target_keys - candidate_keys),
                "unexpected": sorted(candidate_keys - target_keys),
            }
    return best_sd, best_info


def load_state_dict_forgiving(
    module: torch.nn.Module,
    checkpoint: dict[str, Any],
    *,
    strict: bool = False,
    prefixes: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    state_dict = unwrap_state_dict(checkpoint)
    use_prefixes = prefixes or [""]
    candidate, info = _best_state_dict_for_module(module, state_dict, use_prefixes)
    load_result = module.load_state_dict(candidate, strict=strict)
    missing = list(load_result.missing_keys)
    unexpected = list(load_result.unexpected_keys)
    return missing, unexpected


def _stats_candidate_dicts(obj: Any) -> list[dict[str, Any]]:
    if not isinstance(obj, dict):
        return []
    out = [obj]
    for key in ("latent_stats", "stats", "normalization", "normalizer", "metadata"):
        value = obj.get(key)
        if isinstance(value, dict):
            out.append(value)
    return out


def _extract_latent_stats(checkpoint: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor] | None:
    key_aliases = {
        "mean": ("mean", "latent_mean", "feature_mean"),
        "var": ("var", "variance", "latent_var"),
        "std": ("std", "feature_std", "latent_std"),
    }
    for candidate in _stats_candidate_dicts(checkpoint):
        mean = next((candidate.get(k) for k in key_aliases["mean"] if candidate.get(k) is not None), None)
        var = next((candidate.get(k) for k in key_aliases["var"] if candidate.get(k) is not None), None)
        std = next((candidate.get(k) for k in key_aliases["std"] if candidate.get(k) is not None), None)
        if mean is None:
            continue
        mean_t = torch.as_tensor(mean, dtype=torch.float32)
        if var is None:
            if std is None:
                continue
            std_t = torch.as_tensor(std, dtype=torch.float32)
            var_t = std_t.square()
        else:
            var_t = torch.as_tensor(var, dtype=torch.float32)
        return mean_t, var_t
    return None


def _maybe_initialize_latent_stats_from_checkpoint(model: torch.nn.Module, checkpoint: dict[str, Any]) -> bool:
    setter = getattr(model, "set_latent_stats", None)
    if setter is None:
        return False
    stats = _extract_latent_stats(checkpoint)
    if stats is None:
        return False
    mean_t, var_t = stats
    setter(mean_t, var_t)
    return True


def initialize_rae_from_pretrained(
    model: torch.nn.Module,
    config: dict[str, Any],
    *,
    map_location: str = "cpu",
    strict: bool | None = None,
    load_ema: bool | None = None,
) -> dict[str, Any]:
    assets = resolve_rae_assets(config)
    pt_cfg = _get_pretrained_cfg(config)
    strict = bool(pt_cfg.get("strict", False) if strict is None else strict)
    load_ema = bool(pt_cfg.get("load_ema", True) if load_ema is None else load_ema)

    info: dict[str, Any] = {
        "source": assets.source,
        "repo_id": assets.repo_id,
        "revision": assets.revision,
        "repo_root": assets.repo_root,
        "decoder_path": assets.decoder_path,
        "decoder_filename": assets.decoder_filename,
        "autoencoder_checkpoint": assets.autoencoder_checkpoint,
        "generator_path": assets.generator_path,
        "generator_filename": assets.generator_filename,
        "loaded": False,
        "mode": None,
        "missing": [],
        "unexpected": [],
        "latent_stats_loaded": False,
    }

    ckpt_path = assets.autoencoder_checkpoint
    if ckpt_path:
        payload = load_checkpoint(Path(ckpt_path), map_location=map_location)
        if load_ema and isinstance(payload, dict) and "ema_model" in payload:
            payload = {"model": payload["ema_model"], **{k: v for k, v in payload.items() if k != "ema_model"}}
        latent_stats_loaded = _maybe_initialize_latent_stats_from_checkpoint(model, payload) if isinstance(payload, dict) else False
        missing, unexpected = load_state_dict_forgiving(
            model,
            payload,
            strict=strict,
            prefixes=["", "model.", "module.", "autoencoder.", "fae.", "rae."],
        )
        info.update({
            "loaded": True,
            "mode": "autoencoder",
            "missing": missing,
            "unexpected": unexpected,
            "latent_stats_loaded": latent_stats_loaded,
        })
        return info

    decoder_path = assets.decoder_path
    if decoder_path:
        payload = load_checkpoint(Path(decoder_path), map_location=map_location)
        latent_stats_loaded = _maybe_initialize_latent_stats_from_checkpoint(model, payload) if isinstance(payload, dict) else False
        decoder = getattr(model, "decoder", None)
        if decoder is None:
            raise PretrainedAssetLoadError("Model does not expose `.decoder`, so decoder-only weights cannot be loaded.")
        missing, unexpected = load_state_dict_forgiving(
            decoder,
            payload,
            strict=strict,
            prefixes=["", "decoder.", "model.decoder.", "module.decoder.", "autoencoder.decoder.", "rae.decoder."],
        )
        info.update({
            "loaded": True,
            "mode": "decoder",
            "missing": missing,
            "unexpected": unexpected,
            "latent_stats_loaded": latent_stats_loaded,
        })
        return info

    return info
