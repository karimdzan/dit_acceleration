
"""Evaluate a SANA-Sprint depth-pruned student against the original teacher"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from fae.modules.depth_pruning import apply_depth_pruning
from fae.profiling.runtime import RuntimeProfiler, patch_diffusion_pipeline_components, summarize_runs, write_profile_csv


def _dtype_from_string(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype {name!r}; expected fp16, bf16, or fp32.")


def _load_yaml(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _get_nested(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _resolve(args: argparse.Namespace, cfg: dict[str, Any], attr: str, dotted: str, default: Any) -> Any:
    val = getattr(args, attr)
    return val if val is not None else _get_nested(cfg, dotted, default)


def _optional_import_pipeline(class_name: str):
    try:
        import diffusers
    except Exception as exc:
        raise ImportError("diffusers is required. Install with `pip install diffusers transformers accelerate safetensors`.") from exc

    if class_name == "auto":
        for candidate in ("SanaSprintPipeline", "SanaPipeline", "DiffusionPipeline"):
            if hasattr(diffusers, candidate):
                return diffusers, getattr(diffusers, candidate)
        raise ImportError("Could not find SanaSprintPipeline, SanaPipeline, or DiffusionPipeline in diffusers.")

    if not hasattr(diffusers, class_name):
        raise ImportError(f"diffusers has no pipeline class named {class_name!r}.")
    return diffusers, getattr(diffusers, class_name)


def _read_prompts(args: argparse.Namespace, cfg: dict[str, Any], num_samples: int) -> list[str]:
    prompts: list[str] = []
    cfg_prompt = _get_nested(cfg, "generation.prompt", None)
    cfg_prompts = _get_nested(cfg, "generation.prompts", None)
    if cfg_prompt:
        prompts.append(str(cfg_prompt))
    if cfg_prompts:
        prompts.extend([str(x) for x in cfg_prompts])
    if args.prompt:
        prompts.extend(args.prompt)
    prompts_file = args.prompts_file or _get_nested(cfg, "generation.prompts_file", None)
    if prompts_file:
        with open(prompts_file, "r") as f:
            prompts.extend([line.strip() for line in f if line.strip()])
    if not prompts:
        prompts = [
            "a tiny astronaut hatching from an egg on the moon",
            "a cinematic photo of a red fox in a snowy forest",
            "a glass teapot on a wooden table, studio lighting",
            "a small robot painting a watercolor landscape",
        ]
    while len(prompts) < num_samples:
        prompts.extend(prompts)
    return prompts[:num_samples]


def _batch(items: list[str], batch_size: int) -> list[list[str]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _make_generator(device: str, seed: int) -> torch.Generator:
    gen_device = device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    return torch.Generator(device=gen_device).manual_seed(int(seed))


def _sync(device: str) :
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def _write_json(path: Path, payload: Any) :
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _save_images(images: Any, out_dir: Path, start_index: int, overwrite: bool) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for offset, image in enumerate(images):
        path = out_dir / f"{start_index + offset:06d}.png"
        if overwrite or not path.exists():
            image.save(path)
        paths.append(str(path))
    return paths


def _run_generation(
    pipe: Any,
    method_name: str,
    prompts: list[str],
    output_dir: Path,
    batch_size: int,
    steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    seed: int,
    device: str,
    output_type: str,
    max_sequence_length: int,
    max_timesteps: float,
    intermediate_timesteps: float,
    num_images_per_prompt: int,
    use_resolution_binning: bool,
    warmup_batches: int,
    overwrite: bool,
    save_images: bool,
    progress_bar: bool,
) -> dict[str, Any]:
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=not progress_bar)

    profiler = RuntimeProfiler(device=device, use_cuda_events=True)
    patched = patch_diffusion_pipeline_components(profiler, pipe)

    method_dir = output_dir / method_name
    image_dir = method_dir / "images"
    runs = []
    paths = []

    for batch_idx, prompt_batch in enumerate(_batch(prompts, batch_size)):
        profiler.reset()
        generator = _make_generator(device, seed + batch_idx * batch_size)
        _sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = pipe(
                prompt=prompt_batch if len(prompt_batch) > 1 else prompt_batch[0],
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator,
                output_type=output_type,
                return_dict=True,
                max_sequence_length=max_sequence_length,
                max_timesteps=max_timesteps,
                intermediate_timesteps=intermediate_timesteps,
                num_images_per_prompt=num_images_per_prompt,
                use_resolution_binning=use_resolution_binning,
            )
        _sync(device)
        total_ms = (time.perf_counter() - t0) * 1000.0
        components = profiler.snapshot()
        warmup = batch_idx < warmup_batches
        images = getattr(out, "images", None)
        if save_images and images is not None and output_type == "pil":
            paths.extend(_save_images(images, image_dir, batch_idx * batch_size, overwrite=overwrite))

        run_record = {
            "run_index": batch_idx,
            "warmup": warmup,
            "batch_size": len(prompt_batch) * num_images_per_prompt,
            "prompt": prompt_batch,
            "total_wall_ms": total_ms,
            "components": components,
        }
        runs.append(run_record)
        tag = "warmup" if warmup else "measured"
        transformer_ms = components.get("transformer_forward", {}).get("total_wall_ms", 0.0)
        decode_ms = components.get("vae_decode", {}).get("total_wall_ms", 0.0)
        encode_ms = components.get("encode_prompt_total", {}).get("total_wall_ms", 0.0)
        print(
            f"[{method_name}:{tag}] batch {batch_idx + 1}: "
            f"total={total_ms:.2f}ms text={encode_ms:.2f}ms transformer={transformer_ms:.2f}ms decode={decode_ms:.2f}ms"
        )

    profiler.restore()
    summary = summarize_runs(runs, warmup_runs=warmup_batches)
    payload = {"method": method_name, "patched_components": patched, "summary": summary, "runs": runs, "image_paths": paths}
    _write_json(method_dir / "profile.json", payload)
    write_profile_csv(method_dir / "profile_runs.csv", runs, summary)
    return payload


def _make_side_by_side(
    baseline_dir: Path,
    pruned_dir: Path,
    prompts: list[str],
    output_dir: Path,
    n: int,
    columns: int = 2,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_paths = []
    count = min(n, len(prompts))
    for i in range(count):
        b_path = baseline_dir / f"{i:06d}.png"
        p_path = pruned_dir / f"{i:06d}.png"
        if not b_path.exists() or not p_path.exists():
            continue
        b_img = Image.open(b_path).convert("RGB")
        p_img = Image.open(p_path).convert("RGB")
        w, h = b_img.size
        label_h = 48
        canvas = Image.new("RGB", (w * 2, h + label_h), "white")
        canvas.paste(b_img, (0, label_h))
        canvas.paste(p_img, (w, label_h))
        draw = ImageDraw.Draw(canvas)
        short_prompt = prompts[i][:80]
        draw.text((4, 4), "baseline", fill="black")
        draw.text((w + 4, 4), "depth_pruned", fill="black")
        draw.text((4, 22), short_prompt, fill="black")
        pair_path = output_dir / f"pair_{i:04d}.png"
        canvas.save(pair_path)
        pair_paths.append(str(pair_path))

    if not pair_paths:
        return {"pair_paths": [], "grid_path": None}

    pairs = [Image.open(p).convert("RGB") for p in pair_paths]
    pair_w, pair_h = pairs[0].size
    columns = max(1, columns)
    rows = (len(pairs) + columns - 1) // columns
    grid = Image.new("RGB", (pair_w * columns, pair_h * rows), "white")
    for i, img in enumerate(pairs):
        grid.paste(img, ((i % columns) * pair_w, (i // columns) * pair_h))
    grid_path = output_dir / "grid_baseline_vs_depth_pruned.png"
    grid.save(grid_path)
    return {"pair_paths": pair_paths, "grid_path": str(grid_path)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--model-name-or-path", default=None)
    p.add_argument("--pipeline-class", default=None, choices=["auto", "SanaSprintPipeline", "SanaPipeline", "DiffusionPipeline"])
    p.add_argument("--prompt", action="append", default=None)
    p.add_argument("--prompts-file", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-inference-steps", type=int, default=None)
    p.add_argument("--guidance-scale", type=float, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dtype", default=None, choices=["fp16", "bf16", "fp32", "float16", "bfloat16", "float32"])
    p.add_argument("--device", default=None)
    p.add_argument("--output-type", default=None, choices=["pil", "latent", "np"])
    p.add_argument("--max-sequence-length", type=int, default=None)
    p.add_argument("--max-timesteps", type=float, default=None)
    p.add_argument("--intermediate-timesteps", type=float, default=None)
    p.add_argument("--num-images-per-prompt", type=int, default=None)
    p.add_argument("--drop-ratio", type=float, default=None)
    p.add_argument("--drop-count", type=int, default=None)
    p.add_argument("--drop-layers", default=None, help="Comma-separated original layer indices to remove.")
    p.add_argument("--keep-layers", default=None, help="Comma-separated original layer indices to keep.")
    p.add_argument("--prune-strategy", default=None, choices=["uniform", "middle", "early", "late"])
    p.add_argument("--preserve-first", type=int, default=None)
    p.add_argument("--preserve-last", type=int, default=None)
    p.add_argument("--warmup-batches", type=int, default=None)
    p.add_argument("--side-by-side-n", type=int, default=None)
    p.add_argument("--side-by-side-columns", type=int, default=None)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--save-images", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--progress-bar", action="store_true")
    p.add_argument("--no-use-resolution-binning", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = _load_yaml(args.config)

    model_name = _resolve(args, cfg, "model_name_or_path", "model_name_or_path", "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers")
    pipeline_class = _resolve(args, cfg, "pipeline_class", "pipeline_class", "SanaSprintPipeline")
    output_dir = Path(_resolve(args, cfg, "output_dir", "output_dir", "samples/sana_sprint_depth_prune_eval"))
    num_samples = int(_resolve(args, cfg, "num_samples", "generation.num_samples", 16))
    batch_size = int(_resolve(args, cfg, "batch_size", "generation.batch_size", 1))
    steps = int(_resolve(args, cfg, "num_inference_steps", "generation.num_inference_steps", 2))
    guidance_scale = float(_resolve(args, cfg, "guidance_scale", "generation.guidance_scale", 4.5))
    height = int(_resolve(args, cfg, "height", "generation.height", 1024))
    width = int(_resolve(args, cfg, "width", "generation.width", 1024))
    seed = int(_resolve(args, cfg, "seed", "generation.seed", 0))
    dtype_name = str(_resolve(args, cfg, "dtype", "generation.dtype", "bf16"))
    device = str(_resolve(args, cfg, "device", "generation.device", "cuda" if torch.cuda.is_available() else "cpu"))
    output_type = str(_resolve(args, cfg, "output_type", "generation.output_type", "pil"))
    max_sequence_length = int(_resolve(args, cfg, "max_sequence_length", "generation.max_sequence_length", 300))
    max_timesteps = float(_resolve(args, cfg, "max_timesteps", "generation.max_timesteps", 1.5708))
    intermediate_timesteps = float(_resolve(args, cfg, "intermediate_timesteps", "generation.intermediate_timesteps", 1.3))
    num_images_per_prompt = int(_resolve(args, cfg, "num_images_per_prompt", "generation.num_images_per_prompt", 1))
    warmup_batches = int(_resolve(args, cfg, "warmup_batches", "generation.warmup_batches", 1))
    side_by_side_n = int(_resolve(args, cfg, "side_by_side_n", "side_by_side.n", min(16, num_samples)))
    side_by_side_columns = int(_resolve(args, cfg, "side_by_side_columns", "side_by_side.columns", 2))
    use_resolution_binning = not bool(args.no_use_resolution_binning)

    prune_cfg = dict(_get_nested(cfg, "depth_pruning", {}) or {})
    prune_cfg.setdefault("enabled", True)
    prune_cfg.setdefault("mode", "remove")
    if args.drop_ratio is not None:
        prune_cfg["drop_ratio"] = args.drop_ratio
    if args.drop_count is not None:
        prune_cfg["drop_count"] = args.drop_count
    if args.drop_layers is not None:
        prune_cfg["drop_layers"] = args.drop_layers
    if args.keep_layers is not None:
        prune_cfg["keep_layers"] = args.keep_layers
    if args.prune_strategy is not None:
        prune_cfg["strategy"] = args.prune_strategy
    if args.preserve_first is not None:
        prune_cfg["preserve_first"] = args.preserve_first
    if args.preserve_last is not None:
        prune_cfg["preserve_last"] = args.preserve_last

    prompts = _read_prompts(args, cfg, num_samples=num_samples)
    _write_json(output_dir / "prompts.json", {"prompts": prompts})

    _, PipelineClass = _optional_import_pipeline(str(pipeline_class))
    torch_dtype = _dtype_from_string(dtype_name)
    print(f"loading {model_name} as {PipelineClass.__name__} with dtype={torch_dtype} on device={device}")
    pipe = PipelineClass.from_pretrained(model_name, torch_dtype=torch_dtype, local_files_only=bool(args.local_files_only))
    pipe = pipe.to(device)

    if output_type != "pil" and args.save_images:
        raise ValueError("--save-images requires --output-type pil.")

    common_kwargs = dict(
        prompts=prompts,
        output_dir=output_dir,
        batch_size=batch_size,
        steps=steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        seed=seed,
        device=device,
        output_type=output_type,
        max_sequence_length=max_sequence_length,
        max_timesteps=max_timesteps,
        intermediate_timesteps=intermediate_timesteps,
        num_images_per_prompt=num_images_per_prompt,
        use_resolution_binning=use_resolution_binning,
        warmup_batches=warmup_batches,
        overwrite=bool(args.overwrite),
        save_images=True if output_type == "pil" else False,
        progress_bar=bool(args.progress_bar),
    )

    baseline = _run_generation(pipe, method_name="baseline", **common_kwargs)

    pruning_result = apply_depth_pruning(pipe.transformer, prune_cfg)
    print(f"depth pruning: {pruning_result.to_dict()}")

    pruned = _run_generation(pipe, method_name="depth_pruned", **common_kwargs)

    side_by_side = None
    if output_type == "pil":
        side_by_side = _make_side_by_side(
            baseline_dir=output_dir / "baseline" / "images",
            pruned_dir=output_dir / "depth_pruned" / "images",
            prompts=prompts,
            output_dir=output_dir / "side_by_side",
            n=side_by_side_n,
            columns=side_by_side_columns,
        )

    base_total = baseline["summary"]["total_wall_ms"]["mean"]
    pruned_total = pruned["summary"]["total_wall_ms"]["mean"]
    base_transformer = baseline["summary"]["components"].get("transformer_forward", {}).get("mean", 0.0)
    pruned_transformer = pruned["summary"]["components"].get("transformer_forward", {}).get("mean", 0.0)
    comparison = {
        "depth_pruning": pruning_result.to_dict(),
        "total_speedup": base_total / pruned_total if pruned_total else None,
        "transformer_speedup": base_transformer / pruned_transformer if pruned_transformer else None,
        "baseline_total_ms": base_total,
        "depth_pruned_total_ms": pruned_total,
        "baseline_transformer_ms": base_transformer,
        "depth_pruned_transformer_ms": pruned_transformer,
        "side_by_side": side_by_side,
    }
    _write_json(output_dir / "comparison.json", comparison)
    print(f"comparison: {json.dumps(comparison, indent=2)}")
    if side_by_side and side_by_side.get("grid_path"):
        print(f"side-by-side grid: {side_by_side['grid_path']}")


if __name__ == "__main__":
    main()
