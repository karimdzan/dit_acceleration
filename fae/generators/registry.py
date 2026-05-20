from typing import Any

from .common import LatentTensorSpec
from .diffusers_backend import DiffusersBackendConfig, DiffusersTransformerBackend
from .internal import InternalDiTDHBackend, InternalLatentDiTBackend


_REGISTRY = {
    "internal_dit": "internal_dit",
    "internal_dit_dh": "internal_dit_dh",
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
            head_dim=gen_cfg.get("head_dim"),
            use_rope_2d=gen_cfg.get("use_rope_2d", False),
        )
    if name == "internal_dit_dh":
        if bridge_spec is None:
            raise ValueError("bridge_spec is required for internal_dit_dh.")
        return InternalDiTDHBackend(
            spec=bridge_spec,
            hidden_size=tuple(gen_cfg.get("hidden_size", [1152, 2048])),
            depth=tuple(gen_cfg.get("depth", [28, 2])),
            num_heads=tuple(gen_cfg.get("num_heads", [16, 16])),
            mlp_ratio=gen_cfg.get("mlp_ratio", 4.0),
            num_classes=gen_cfg.get("num_classes", config.get("conditioning", {}).get("num_classes", 1000)),
            class_dropout_prob=gen_cfg.get("class_dropout_prob", 0.1),
            use_qknorm=gen_cfg.get("use_qknorm", False),
            use_swiglu=gen_cfg.get("use_swiglu", True),
            use_rope=gen_cfg.get("use_rope", True),
            use_rmsnorm=gen_cfg.get("use_rmsnorm", True),
            wo_shift=gen_cfg.get("wo_shift", False),
            use_pos_embed=gen_cfg.get("use_pos_embed", True),
            objective=gen_cfg.get("objective", "linear_velocity"),
            prediction_type=gen_cfg.get("prediction_type", "velocity"),
            time_dist_type=gen_cfg.get("time_dist_type", config.get("transport", {}).get("time_dist_type", "logit-normal_0_1")),
            loss_weight=gen_cfg.get("loss_weight", config.get("transport", {}).get("loss_weight")),
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
