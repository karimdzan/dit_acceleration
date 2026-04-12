from typing import Any

from .common import LatentTensorSpec
from .diffusers_backend import DiffusersBackendConfig, DiffusersTransformerBackend
from .internal import InternalLatentDiTBackend


_REGISTRY = {
    "internal_dit": "internal_dit",
    "diffusers_dit": "diffusers_dit",
    "diffusers_sd3": "diffusers_sd3",
    "diffusers_sana": "diffusers_sana",
}


def list_generator_backends() -> list[str]:
    return sorted(_REGISTRY)


def build_generator_backend(config: dict[str, Any], bridge_spec: LatentTensorSpec | None = None):
    gen_cfg = config["generator"]
    name = gen_cfg["name"]
    if name == "internal_dit":
        if bridge_spec is None:
            raise ValueError("bridge_spec is required for internal_dit.")
        return InternalLatentDiTBackend(
            spec=bridge_spec,
            model_dim=gen_cfg.get("model_dim", 768),
            depth=gen_cfg.get("depth", 12),
            num_heads=gen_cfg.get("num_heads", 12),
            cond_dim=gen_cfg.get("cond_dim", 1024),
            mlp_ratio=gen_cfg.get("mlp_ratio", 4.0),
            objective=gen_cfg.get("objective", "diffusion"),
            prediction_type=gen_cfg.get("prediction_type", "v_prediction"),
            time_shift=gen_cfg.get("time_shift", 0.0),
        )
    if name in {"diffusers_dit", "diffusers_sd3", "diffusers_sana"}:
        return DiffusersTransformerBackend(
            DiffusersBackendConfig(
                kind=name,
                transformer_name_or_path=gen_cfg.get("transformer_name_or_path"),
                pipeline_name_or_path=gen_cfg.get("pipeline_name_or_path"),
                pipeline_class=gen_cfg.get("pipeline_class"),
                transformer_subfolder=gen_cfg.get("transformer_subfolder", "transformer"),
                torch_dtype=gen_cfg.get("torch_dtype", "bfloat16"),
                objective=gen_cfg.get("objective", "diffusion"),
                prediction_type=gen_cfg.get("prediction_type", "epsilon"),
                train_mode=gen_cfg.get("train_mode", "frozen"),
                guidance_dropout=gen_cfg.get("guidance_dropout", 0.0),
                sample_size=gen_cfg.get("sample_size"),
                in_channels=gen_cfg.get("in_channels"),
            )
        )
    raise KeyError(f"Unknown generator backend: {name}. Available: {list_generator_backends()}")
