import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

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


def _read_prompts(args: argparse.Namespace, cfg: dict[str, Any]) -> list[str]:
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
        prompts = ["a tiny astronaut hatching from an egg on the moon"]
    return prompts


def _batch_prompts(prompts: list[str], batch_size: int) -> list[list[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    return [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]


def _make_generator(device: str, seed: int) -> torch.Generator:
    gen_device = device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    return torch.Generator(device=gen_device).manual_seed(int(seed))


def _sync(device: str):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def _write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _save_outputs(images: Any, out_dir: Path, run_index: int) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    if images is None:
        return paths
    if isinstance(images, torch.Tensor):
        path = out_dir / f"run_{run_index:04d}_latents.pt"
        torch.save(images.detach().cpu(), path)
        return [str(path)]
    for i, image in enumerate(images):
        if hasattr(image, "save"):
            path = out_dir / f"run_{run_index:04d}_{i:02d}.png"
            image.save(path)
            paths.append(str(path))
    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--model-name-or-path", default=None)
    p.add_argument("--pipeline-class", default=None, choices=["auto", "SanaSprintPipeline", "SanaPipeline", "DiffusionPipeline"])
    p.add_argument("--prompt", action="append", default=None, help="Prompt. Can be passed multiple times.")
    p.add_argument("--prompts-file", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--num-runs", type=int, default=None)
    p.add_argument("--warmup-runs", type=int, default=None)
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
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--save-images", action="store_true")
    p.add_argument("--enable-vae-tiling", action="store_true")
    p.add_argument("--enable-vae-slicing", action="store_true")
    p.add_argument("--disable-progress-bar", action="store_true")
    p.add_argument("--no-use-resolution-binning", action="store_true")
    p.add_argument("--no-cuda-events", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = _load_yaml(args.config)

    model_name = _resolve(args, cfg, "model_name_or_path", "model_name_or_path", "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers")
    pipeline_class = _resolve(args, cfg, "pipeline_class", "pipeline_class", "SanaSprintPipeline")
    output_dir = Path(_resolve(args, cfg, "output_dir", "output_dir", "samples/sana_sprint_profile"))
    num_runs = int(_resolve(args, cfg, "num_runs", "profile.num_runs", 8))
    warmup_runs = int(_resolve(args, cfg, "warmup_runs", "profile.warmup_runs", 2))
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
    max_timesteps = float(_resolve(args, cfg, "max_timesteps", "generation.max_timesteps", 1.57080))
    intermediate_timesteps = float(_resolve(args, cfg, "intermediate_timesteps", "generation.intermediate_timesteps", 1.3))
    num_images_per_prompt = int(_resolve(args, cfg, "num_images_per_prompt", "generation.num_images_per_prompt", 1))
    use_resolution_binning = not bool(args.no_use_resolution_binning)

    prompts = _read_prompts(args, cfg)
    prompt_batches = _batch_prompts(prompts, batch_size)
    if len(prompt_batches) == 1 and num_runs > 1:
        prompt_batches = prompt_batches * num_runs
    elif len(prompt_batches) < num_runs:
        prompt_batches = (prompt_batches * ((num_runs + len(prompt_batches) - 1) // len(prompt_batches)))[:num_runs]
    else:
        prompt_batches = prompt_batches[:num_runs]

    diffusers, PipelineClass = _optional_import_pipeline(str(pipeline_class))
    torch_dtype = _dtype_from_string(dtype_name)

    print(f"loading {model_name} as {PipelineClass.__name__} with dtype={torch_dtype} on device={device}")
    pipe = PipelineClass.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        local_files_only=bool(args.local_files_only),
    )
    pipe = pipe.to(device)

    if args.disable_progress_bar and hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    if args.enable_vae_tiling and hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    if args.enable_vae_slicing and hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()

    profiler = RuntimeProfiler(device=device, use_cuda_events=not bool(args.no_cuda_events))
    patched = patch_diffusion_pipeline_components(profiler, pipe)
    print(f"patched components: {patched}")

    output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    generated_paths: list[str] = []

    for run_idx, prompt_batch in enumerate(prompt_batches):
        profiler.reset()
        generator = _make_generator(device, seed + run_idx)
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
        warmup = run_idx < warmup_runs

        images = getattr(out, "images", None)
        if args.save_images and not warmup:
            generated_paths.extend(_save_outputs(images, output_dir / "images", run_idx))

        run_record = {
            "run_index": run_idx,
            "warmup": warmup,
            "batch_size": len(prompt_batch) * num_images_per_prompt,
            "prompt": prompt_batch,
            "total_wall_ms": total_ms,
            "components": components,
        }
        runs.append(run_record)

        measured_tag = "warmup" if warmup else "measured"
        transformer_ms = components.get("transformer_forward", {}).get("total_wall_ms", 0.0)
        decode_ms = components.get("vae_decode", {}).get("total_wall_ms", 0.0)
        encode_ms = components.get("encode_prompt_total", {}).get("total_wall_ms", 0.0)
        print(
            f"[{measured_tag}] run {run_idx + 1}/{len(prompt_batches)}: "
            f"total={total_ms:.2f}ms text={encode_ms:.2f}ms transformer={transformer_ms:.2f}ms decode={decode_ms:.2f}ms"
        )

    profiler.restore()
    summary = summarize_runs(runs, warmup_runs=warmup_runs)

    payload = {
        "config": {
            "model_name_or_path": model_name,
            "pipeline_class": str(pipeline_class),
            "num_runs": num_runs,
            "warmup_runs": warmup_runs,
            "batch_size": batch_size,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "dtype": dtype_name,
            "device": device,
            "output_type": output_type,
            "use_resolution_binning": use_resolution_binning,
            "patched_components": patched,
        },
        "summary": summary,
        "runs": runs,
        "generated_paths": generated_paths,
    }

    _write_json(output_dir / "profile.json", payload)
    write_profile_csv(output_dir / "profile_runs.csv", runs, summary)
    print(f"wrote {output_dir / 'profile.json'}")
    print(f"wrote {output_dir / 'profile_runs.csv'}")

    print("\nMeasured mean wall time:")
    print(f"  total:       {summary['total_wall_ms']['mean']:.2f} ms")
    for name, comp in summary["components"].items():
        print(f"  {name:22s} {comp['mean']:.2f} ms  ({comp['mean_percent_of_total_wall']:.1f}%)")


if __name__ == "__main__":
    main()
