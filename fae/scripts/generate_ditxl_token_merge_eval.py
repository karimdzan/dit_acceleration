"""Generate and evaluate baseline vs token-merged DiT-XL samples"""

import argparse
import time
from pathlib import Path
from typing import Any

import torch

from fae.evaluation.ditxl_eval_utils import (
    build_side_by_side_outputs,
    ensure_dir,
    get_nested,
    image_path_for_index,
    load_yaml_config,
    make_class_labels,
    make_id_to_label,
    write_json,
)
from fae.evaluation.image_folder_metrics import calculate_fid_is_with_torch_fidelity
from fae.modules.token_merging import apply_token_merging, iter_token_merging_wrappers


def _optional_import_diffusers():
    try:
        import diffusers
        from diffusers import DiTPipeline
    except Exception as exc:
        raise ImportError(
            "diffusers is required for this script. Install with `pip install diffusers accelerate safetensors`."
        ) from exc
    return diffusers, DiTPipeline


def _dtype_from_string(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype {name!r}; expected fp16, bf16, or fp32.")


def _resolve(args: argparse.Namespace, config: dict[str, Any], name: str, default: Any, dotted: str | None = None) -> Any:
    value = getattr(args, name)
    if value is not None:
        return value
    return get_nested(config, dotted or name.replace("_", "-"), default)


def _cuda_synchronize(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _configure_scheduler(pipe: Any, scheduler_name: str, diffusers: Any):
    scheduler_name = scheduler_name.lower()
    if scheduler_name in {"native", "default", "ddim"}:
        return
    if scheduler_name in {"dpm", "dpm_solver", "dpmsolver"}:
        scheduler_cls = getattr(diffusers, "DPMSolverMultistepScheduler")
        pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
        return
    raise ValueError(f"Unsupported scheduler {scheduler_name!r}; expected native/ddim or dpm.")


def _make_generator(device: str, seed: int) -> torch.Generator:
    gen_device = device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    return torch.Generator(device=gen_device).manual_seed(int(seed))


def _save_images(images: list[Any], labels: list[int], start_index: int, out_dir: Path, overwrite: bool) -> list[Path]:
    paths: list[Path] = []
    for offset, image in enumerate(images):
        idx = start_index + offset
        label = int(labels[offset])
        path = image_path_for_index(out_dir, idx, label)
        if overwrite or not path.exists():
            image.save(path)
        paths.append(path)
    return paths


def _generate_method(
    pipe: Any,
    method_name: str,
    labels: list[int],
    output_dir: Path,
    batch_size: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    device: str,
    overwrite: bool,
    disable_progress_bar: bool,
    warmup_batches: int = 1,
) -> dict[str, Any]:
    method_dir = ensure_dir(output_dir / method_name / "images")
    batch_timings: list[dict[str, Any]] = []
    generated = 0
    skipped = 0
    total_start = time.perf_counter()

    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=disable_progress_bar)

    for start in range(0, len(labels), batch_size):
        batch_labels = labels[start : start + batch_size]
        expected_paths = [image_path_for_index(method_dir, start + i, int(label)) for i, label in enumerate(batch_labels)]
        if not overwrite and all(path.exists() for path in expected_paths):
            skipped += len(batch_labels)
            continue

        generator = _make_generator(device, seed + start)
        _cuda_synchronize(device)
        t0 = time.perf_counter()
        output = pipe(
            class_labels=[int(x) for x in batch_labels],
            guidance_scale=float(guidance_scale),
            generator=generator,
            num_inference_steps=int(num_inference_steps),
            output_type="pil",
            return_dict=True,
        )
        _cuda_synchronize(device)
        elapsed = time.perf_counter() - t0
        images = list(output.images)
        _save_images(images, batch_labels, start, method_dir, overwrite=overwrite)
        generated += len(images)
        batch_timings.append(
            {
                "start_index": start,
                "batch_size": len(images),
                "seconds": elapsed,
                "seconds_per_image": elapsed / max(1, len(images)),
            }
        )
        print(
            f"[{method_name}] batch {start // batch_size + 1}: "
            f"{len(images)} images in {elapsed:.3f}s ({elapsed / max(1, len(images)):.3f}s/img)"
        )

    total_seconds = time.perf_counter() - total_start
    timed_images = sum(item["batch_size"] for item in batch_timings)
    timed_seconds = sum(item["seconds"] for item in batch_timings)
    steady_batches = batch_timings[max(0, int(warmup_batches)) :]
    steady_images = sum(item["batch_size"] for item in steady_batches)
    steady_seconds = sum(item["seconds"] for item in steady_batches)
    summary = {
        "method": method_name,
        "image_dir": str(method_dir),
        "requested_images": len(labels),
        "generated_images_this_run": generated,
        "skipped_existing_images": skipped,
        "total_wall_seconds": total_seconds,
        "timed_generation_seconds": timed_seconds,
        "timed_images": timed_images,
        "seconds_per_image": timed_seconds / timed_images if timed_images else None,
        "images_per_second": timed_images / timed_seconds if timed_seconds else None,
        "warmup_batches_excluded_from_steady_state": int(warmup_batches),
        "steady_state_seconds": steady_seconds,
        "steady_state_images": steady_images,
        "steady_state_seconds_per_image": steady_seconds / steady_images if steady_images else None,
        "steady_state_images_per_second": steady_images / steady_seconds if steady_seconds else None,
        "batch_timings": batch_timings,
    }
    write_json(output_dir / method_name / "generation_summary.json", summary)
    return summary


def _token_merge_runtime_summary(pipe: Any) -> dict[str, Any]:
    wrappers = list(iter_token_merging_wrappers(pipe.transformer))
    last_stats = []
    for idx, wrapper in enumerate(wrappers):
        stats = getattr(wrapper, "last_stats", None)
        if stats is None:
            continue
        item = {
            "wrapper_index": idx,
            "calls": getattr(wrapper, "calls", None),
            "original_tokens": getattr(stats, "original_tokens", None),
            "merged_tokens": getattr(stats, "merged_tokens", None),
            "keep_ratio": getattr(stats, "keep_ratio", None),
        }
        if hasattr(stats, "grid_size"):
            item["grid_size"] = list(stats.grid_size)
            item["merged_grid_size"] = list(stats.merged_grid_size)
        if hasattr(stats, "num_tiles"):
            item["num_tiles"] = stats.num_tiles
            item["dst_per_tile"] = stats.dst_per_tile
        last_stats.append(item)
    controller = getattr(pipe.transformer, "token_merging_controller", None)
    return {
        "num_wrappers": len(wrappers),
        "wrapped_layers": list(getattr(pipe.transformer, "token_merging_layers", [])),
        "last_stats": last_stats,
        "controller": None if controller is None else {
            "step_index": getattr(controller, "step_index", None),
            "plan_recomputes": getattr(controller, "plan_recomputes", None),
            "reuse_across_layers": getattr(controller, "reuse_across_layers", None),
            "recompute_steps": getattr(controller, "recompute_steps", None),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Optional YAML config. CLI args override config values.")
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dtype", default=None, choices=["fp16", "bf16", "fp32", "float16", "bfloat16", "float32"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--scheduler", default=None, choices=["native", "ddim", "dpm", "dpm_solver", "dpmsolver"])
    parser.add_argument("--label-mode", default=None, choices=["random", "round_robin", "labels_file"])
    parser.add_argument("--labels-file", default=None)
    parser.add_argument("--side-by-side-n", type=int, default=None)
    parser.add_argument("--side-by-side-columns", type=int, default=None)
    parser.add_argument("--imagenet-val-dir", default=None, help="Optional real ImageNet validation image folder for FID/IS.")
    parser.add_argument("--compute-metrics", action="store_true", help="Compute FID/IS after generation if --imagenet-val-dir is set.")
    parser.add_argument("--metric-batch-size", type=int, default=None)
    parser.add_argument("--metric-num-workers", type=int, default=None)
    parser.add_argument("--metric-cpu", action="store_true")
    parser.add_argument("--kid", action="store_true", help="Also compute KID with torch-fidelity.")
    parser.add_argument("--token-merge-method", default=None, choices=["toma", "spatial_pool"])
    parser.add_argument("--token-merge-layers", default=None, help="Comma-separated explicit layer indices.")
    parser.add_argument("--token-merge-ratio", type=float, default=None)
    parser.add_argument("--token-merge-num-tiles", type=int, default=None)
    parser.add_argument("--token-merge-attention-scale", type=float, default=None)
    parser.add_argument("--token-merge-recompute-steps", type=int, default=None)
    parser.add_argument("--token-merge-facility-batch", default=None, choices=["first", "all"])
    parser.add_argument("--token-merge-stride", type=int, default=None)
    parser.add_argument("--warmup-batches", type=int, default=None, help="Exclude this many first generated batches from steady-state timing summaries.")
    parser.add_argument("--token-merge-start-layer", type=int, default=None)
    parser.add_argument("--token-merge-end-layer", type=int, default=None)
    parser.add_argument("--token-merge-every", type=int, default=None)
    parser.add_argument("--token-merge-min-image-tokens", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--enable-xformers", action="store_true")
    parser.add_argument("--progress-bar", action="store_true")
    return parser.parse_args()


def main() :
    args = parse_args()
    config = load_yaml_config(args.config)

    model_name = _resolve(args, config, "model_name_or_path", "facebook/DiT-XL-2-256", "model_name_or_path")
    output_dir = ensure_dir(_resolve(args, config, "output_dir", "samples/ditxl_token_merge_eval", "output_dir"))
    num_samples = int(_resolve(args, config, "num_samples", 128, "generation.num_samples"))
    batch_size = int(_resolve(args, config, "batch_size", 8, "generation.batch_size"))
    steps = int(_resolve(args, config, "num_inference_steps", 50, "generation.num_inference_steps"))
    guidance = float(_resolve(args, config, "guidance_scale", 4.0, "generation.guidance_scale"))
    seed = int(_resolve(args, config, "seed", 0, "generation.seed"))
    dtype_name = str(_resolve(args, config, "dtype", "fp16", "generation.dtype"))
    device = str(_resolve(args, config, "device", "cuda" if torch.cuda.is_available() else "cpu", "generation.device"))
    scheduler = str(_resolve(args, config, "scheduler", "dpm", "generation.scheduler"))
    label_mode = str(_resolve(args, config, "label_mode", "random", "generation.label_mode"))
    labels_file = _resolve(args, config, "labels_file", None, "generation.labels_file")
    side_by_side_n = int(_resolve(args, config, "side_by_side_n", 16, "side_by_side.n"))
    side_by_side_cols = int(_resolve(args, config, "side_by_side_columns", 2, "side_by_side.columns"))
    metric_batch_size = int(_resolve(args, config, "metric_batch_size", 64, "metrics.batch_size"))
    metric_num_workers = int(_resolve(args, config, "metric_num_workers", 4, "metrics.num_workers"))
    warmup_batches = int(_resolve(args, config, "warmup_batches", 1, "generation.warmup_batches"))

    token_cfg = dict(get_nested(config, "token_merging", {}) or {})
    token_cfg.setdefault("enabled", True)
    token_cfg.setdefault("method", "toma")
    if args.token_merge_method is not None:
        token_cfg["method"] = args.token_merge_method
    if args.token_merge_layers is not None:
        token_cfg["layers"] = [int(x.strip()) for x in args.token_merge_layers.split(",") if x.strip()]
    token_cfg["stride"] = int(args.token_merge_stride if args.token_merge_stride is not None else token_cfg.get("stride", 2))
    token_cfg["start_layer"] = int(
        args.token_merge_start_layer if args.token_merge_start_layer is not None else token_cfg.get("start_layer", 7)
    )
    token_cfg["end_layer"] = int(
        args.token_merge_end_layer if args.token_merge_end_layer is not None else token_cfg.get("end_layer", 21)
    )
    token_cfg["every"] = int(args.token_merge_every if args.token_merge_every is not None else token_cfg.get("every", 1))
    token_cfg["min_image_tokens"] = int(
        args.token_merge_min_image_tokens
        if args.token_merge_min_image_tokens is not None
        else token_cfg.get("min_image_tokens", 64)
    )
    token_cfg["ratio"] = float(args.token_merge_ratio if args.token_merge_ratio is not None else token_cfg.get("ratio", 0.25))
    token_cfg["num_tiles"] = int(args.token_merge_num_tiles if args.token_merge_num_tiles is not None else token_cfg.get("num_tiles", 16))
    token_cfg["attention_scale"] = float(
        args.token_merge_attention_scale if args.token_merge_attention_scale is not None else token_cfg.get("attention_scale", 1000.0)
    )
    token_cfg["recompute_steps"] = int(
        args.token_merge_recompute_steps if args.token_merge_recompute_steps is not None else token_cfg.get("recompute_steps", 1)
    )
    if args.token_merge_facility_batch is not None:
        token_cfg["facility_batch"] = args.token_merge_facility_batch
    token_cfg.setdefault("facility_batch", "first")
    token_cfg.setdefault("reuse_across_layers", True)
    token_cfg.setdefault("num_prefix_tokens", 0)

    labels = make_class_labels(num_samples, mode=label_mode, seed=seed, labels_file=labels_file)
    write_json(output_dir / "class_labels.json", {"labels": labels, "mode": label_mode, "seed": seed})

    diffusers, DiTPipeline = _optional_import_diffusers()
    torch_dtype = _dtype_from_string(dtype_name)
    print(f"loading {model_name} with dtype={torch_dtype} on device={device}")
    pipe = DiTPipeline.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        local_files_only=bool(args.local_files_only),
    )
    _configure_scheduler(pipe, scheduler, diffusers)
    pipe = pipe.to(device)
    if args.enable_xformers and hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        pipe.enable_xformers_memory_efficient_attention()

    id_to_label = make_id_to_label(getattr(pipe, "labels", None))

    metadata: dict[str, Any] = {
        "model_name_or_path": model_name,
        "scheduler": scheduler,
        "num_samples": num_samples,
        "batch_size": batch_size,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "seed": seed,
        "dtype": dtype_name,
        "device": device,
        "warmup_batches": warmup_batches,
        "token_merging": token_cfg,
    }

    baseline_summary = _generate_method(
        pipe=pipe,
        method_name="baseline",
        labels=labels,
        output_dir=output_dir,
        batch_size=batch_size,
        num_inference_steps=steps,
        guidance_scale=guidance,
        seed=seed,
        device=device,
        overwrite=bool(args.overwrite),
        disable_progress_bar=not bool(args.progress_bar),
        warmup_batches=warmup_batches,
    )
    metadata["baseline"] = baseline_summary

    wrapped_layers = apply_token_merging(pipe.transformer, token_cfg)
    print(f"token merging wrapped layers: {wrapped_layers}")
    token_merge_summary = _generate_method(
        pipe=pipe,
        method_name="token_merge",
        labels=labels,
        output_dir=output_dir,
        batch_size=batch_size,
        num_inference_steps=steps,
        guidance_scale=guidance,
        seed=seed,
        device=device,
        overwrite=bool(args.overwrite),
        disable_progress_bar=not bool(args.progress_bar),
        warmup_batches=warmup_batches,
    )
    metadata["token_merge"] = token_merge_summary
    metadata["token_merge_runtime"] = _token_merge_runtime_summary(pipe)

    side_by_side = build_side_by_side_outputs(
        baseline_dir=Path(baseline_summary["image_dir"]),
        token_merge_dir=Path(token_merge_summary["image_dir"]),
        output_dir=output_dir / "side_by_side",
        labels=labels,
        n=side_by_side_n,
        id_to_label=id_to_label,
        grid_columns=side_by_side_cols,
    )
    metadata["side_by_side"] = side_by_side

    metrics = None
    if args.compute_metrics or args.imagenet_val_dir:
        if not args.imagenet_val_dir:
            raise ValueError("--compute-metrics requires --imagenet-val-dir.")
        metrics = {}
        for method_name, summary in (("baseline", baseline_summary), ("token_merge", token_merge_summary)):
            print(f"computing FID/IS for {method_name} against {args.imagenet_val_dir}")
            metrics[method_name] = calculate_fid_is_with_torch_fidelity(
                real_dir=args.imagenet_val_dir,
                fake_dir=summary["image_dir"],
                batch_size=metric_batch_size,
                cuda=not args.metric_cpu,
                num_workers=metric_num_workers,
                verbose=True,
                compute_kid=bool(args.kid),
            )
        write_json(output_dir / "metrics.json", metrics)
        metadata["metrics"] = metrics

    write_json(output_dir / "run_metadata.json", metadata)
    print(f"wrote metadata to {output_dir / 'run_metadata.json'}")
    if side_by_side.get("grid_path"):
        print(f"side-by-side grid: {side_by_side['grid_path']}")
    if metrics is not None:
        print(f"metrics: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
