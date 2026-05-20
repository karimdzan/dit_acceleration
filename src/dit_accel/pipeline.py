import os

import torch
from diffusers import SanaSprintPipeline, SanaTransformer2DModel


DEFAULT_CKPT = os.environ.get(
    "SANA_SPRINT_PATH",
    "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
)


def load_pipeline(
    variant: str = "bf16",
    checkpoint: str = DEFAULT_CKPT,
    device: str | torch.device = "cuda",
    *,
    cache_schedule: dict | None = None,
    state_cache_threshold: float = 0.0,
    sana_ffn_sparse_config: object | None = None,
    sparsity_path: str | None = None,
    sparse_ffn_use_kernel: bool = False,
) -> SanaSprintPipeline:
    """Build a Sana Sprint pipeline configured for the given variant flags."""
    flags = set(variant.split("+"))
    if "cached" in flags:
        flags.discard("cached")
        flags.update({"xattn", "lacache"})

    dtype = torch.bfloat16
    transformer_kwargs: dict = {"torch_dtype": dtype, "subfolder": "transformer"}

    use_int8 = "int8" in flags
    use_int8_bnb = "int8_bnb" in flags
    if use_int8_bnb:
        from diffusers import BitsAndBytesConfig as DiffusersBnb
        transformer_kwargs["quantization_config"] = DiffusersBnb(load_in_8bit=True)

    transformer = SanaTransformer2DModel.from_pretrained(checkpoint, **transformer_kwargs)

    if use_int8 and not use_int8_bnb:
        transformer = transformer.to(device)
        from .quant_compat import torchao_int8_weight_only_config, quantize_
        quantize_(transformer, torchao_int8_weight_only_config())

    pipe = SanaSprintPipeline.from_pretrained(
        checkpoint,
        transformer=transformer,
        torch_dtype=dtype,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    if "block_cache" in flags:
        from .caching.block_feature_cache import install_block_feature_cache
        install_block_feature_cache(pipe, schedule=cache_schedule)

    if "xattn" in flags:
        from .caching.cross_attn_cache import install_cross_attn_cache
        install_cross_attn_cache(pipe)

    if "lacache" in flags:
        from .caching.linear_attn_state_cache import install_state_cache
        install_state_cache(
            pipe,
            schedule=cache_schedule,
            auto_threshold=state_cache_threshold,
        )

    if "sparse_ffn" in flags:
        if not sparsity_path:
            raise ValueError("variant 'sparse_ffn' requires a sparsity_path.")
        from .sparsity.mixffn_sparsifier import install_mixffn_sparsifier
        payload = torch.load(sparsity_path, map_location="cpu", weights_only=False)
        thresholds = payload.get("thresholds", payload) if isinstance(payload, dict) else payload
        install_mixffn_sparsifier(pipe, thresholds=thresholds, use_kernel=sparse_ffn_use_kernel)

    if "gsparse" in flags or "sparseffn" in flags:
        from .sparsity.sana_sparse_ffn import SanaFFNGroupSparseConfig, install_sana_ffn_group_sparse
        if sana_ffn_sparse_config is None:
            sana_ffn_sparse_config = SanaFFNGroupSparseConfig(mode="dynamic", keep_ratio=0.90)
        install_sana_ffn_group_sparse(pipe, sana_ffn_sparse_config)

    return pipe


def reset_caches(pipe) -> None:
    """Wipe per-generation cache state. Cumulative counters are preserved."""
    if hasattr(pipe, "_dit_accel_xattn_cache"):
        pipe._dit_accel_xattn_cache.clear()
    if hasattr(pipe, "_dit_accel_state_cache"):
        pipe._dit_accel_state_cache.clear()
    if hasattr(pipe, "_dit_accel_block_cache"):
        pipe._dit_accel_block_cache.clear_for_generation()
    if hasattr(pipe, "_dit_accel_sparsity"):
        pipe._dit_accel_sparsity.flush()
