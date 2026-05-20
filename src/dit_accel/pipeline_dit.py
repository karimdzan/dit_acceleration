import os

import torch
from diffusers import DiTPipeline, Transformer2DModel


DEFAULT_DIT_CKPT = os.environ.get("DIT_XL_PATH", "facebook/DiT-XL-2-256")


def load_dit_pipeline(
    variant: str = "bf16",
    checkpoint: str = DEFAULT_DIT_CKPT,
    device: str | torch.device = "cuda",
    *,
    cache_schedule: dict | None = None,
    sparsity_path: str | None = None,
) -> DiTPipeline:
    """Build a DiT-XL pipeline configured for the given variant flags."""
    flags = set(variant.split("+"))
    dtype = torch.bfloat16

    transformer_kwargs: dict = {"torch_dtype": dtype, "subfolder": "transformer"}

    use_int8 = "int8" in flags
    use_int8_bnb = "int8_bnb" in flags
    if use_int8_bnb:
        from diffusers import BitsAndBytesConfig as DiffusersBnb
        transformer_kwargs["quantization_config"] = DiffusersBnb(load_in_8bit=True)

    transformer = Transformer2DModel.from_pretrained(checkpoint, **transformer_kwargs)

    if use_int8 and not use_int8_bnb:
        transformer = transformer.to(device)
        from .quant_compat import torchao_int8_weight_only_config, quantize_
        quantize_(transformer, torchao_int8_weight_only_config())

    pipe = DiTPipeline.from_pretrained(
        checkpoint,
        transformer=transformer,
        torch_dtype=dtype,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    if os.environ.get("DIT_ACCEL_COMPILE", "0") == "1":
        pipe.transformer = torch.compile(
            pipe.transformer,
            mode="reduce-overhead",
            fullgraph=False,
        )

    if "block_cache" in flags:
        from .caching.block_feature_cache import install_block_feature_cache
        install_block_feature_cache(pipe, schedule=cache_schedule)

    if "sparse_ffn" in flags:
        if not sparsity_path:
            raise ValueError("variant 'sparse_ffn' requires a sparsity_path.")
        from .sparsity.dit_mlp_sparsifier import install_dit_mlp_sparsifier
        payload = torch.load(sparsity_path, map_location="cpu", weights_only=False)
        thresholds = payload.get("thresholds", payload) if isinstance(payload, dict) else payload
        install_dit_mlp_sparsifier(pipe, thresholds=thresholds)

    return pipe


def reset_dit_caches(pipe: DiTPipeline) -> None:
    """Wipe per-generation cache state. Cumulative counters are preserved."""
    if hasattr(pipe, "_dit_accel_block_cache"):
        pipe._dit_accel_block_cache.clear_for_generation()
    if hasattr(pipe, "_dit_accel_sparsity"):
        pipe._dit_accel_sparsity.flush()
