"""Generate samples for a variant (DDP via torchrun)."""
import json
import os
import time
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from dit_accel.data import load_coco_30k_captions
from dit_accel.evaluation.imagenet_class_ids import build_class_ids
from dit_accel.evaluation.imagenet_prompts import build_prompts
from dit_accel.evaluation.latency import LatencyTimer, peak_memory_gb, reset_peak_memory

from ._common import CONFIG_PATH


def _load_schedule(cfg: DictConfig, variant: str, rank: int):
    if cfg.schedule_path is None or cfg.k_per_step <= 0:
        return None
    schedule_path = Path(cfg.schedule_path)
    if not schedule_path.exists():
        return None
    if "block_cache" in variant:
        from dit_accel.caching.block_feature_cache import build_greedy_schedule
        payload = torch.load(schedule_path)
        schedule = build_greedy_schedule(payload["deltas_by_step"], cfg.k_per_step)
    else:
        from dit_accel.caching.linear_attn_state_cache import build_greedy_schedule
        schedule = build_greedy_schedule(schedule_path, cfg.k_per_step)
    if rank == 0:
        print(f"Loaded schedule from {schedule_path} (k={cfg.k_per_step}, {len(schedule)} entries)")
    return schedule


def _build_pipe(cfg: DictConfig, device: str, schedule, rank: int):
    variant = cfg.variant.name
    if cfg.model.name == "sana_sprint":
        from dit_accel.pipeline import load_pipeline
        sana_ffn_sparse_config = None
        if "gsparse" in variant or "sparseffn" in variant:
            from dit_accel.sparsity.sana_sparse_ffn import SanaFFNGroupSparseConfig
            ffn_cfg = cfg.get("sana_ffn") or {}
            sana_ffn_sparse_config = SanaFFNGroupSparseConfig(
                mode=ffn_cfg.get("mode", "static"),
                keep_ratio=float(ffn_cfg.get("keep_ratio", 0.90)),
                group_size=int(ffn_cfg.get("group_size", 32)),
                min_skip_ratio=float(ffn_cfg.get("min_skip_ratio", 0.05)),
                plan_path=ffn_cfg.get("plan_path"),
                verbose=(rank == 0),
            )
        return load_pipeline(
            variant=variant,
            checkpoint=cfg.model.checkpoint,
            device=device,
            cache_schedule=schedule,
            state_cache_threshold=float(cfg.variant.get("state_cache_threshold", 0.0)),
            sana_ffn_sparse_config=sana_ffn_sparse_config,
            sparsity_path=cfg.sparsity_path,
            sparse_ffn_use_kernel=bool(cfg.variant.get("use_kernel", False)),
        )
    from dit_accel.pipeline_dit import load_dit_pipeline
    return load_dit_pipeline(
        variant=variant,
        checkpoint=cfg.model.checkpoint,
        device=device,
        cache_schedule=schedule,
        sparsity_path=cfg.sparsity_path,
    )


def _conditioning(cfg: DictConfig):
    if cfg.dataset.name == "coco30k":
        if cfg.model.name == "dit_xl":
            raise SystemExit("DiT-XL is class-conditional; use dataset=imagenet.")
        items = load_coco_30k_captions(cfg.dataset.coco_local_dir)
        if len(items) != 30000:
            print(f"warning: expected 30000 COCO captions, got {len(items)}")
        return items
    if cfg.model.name == "sana_sprint":
        return build_prompts(samples_per_class=cfg.samples_per_class)
    return build_class_ids(samples_per_class=cfg.samples_per_class)


def _reset(pipe, model_name: str):
    if model_name == "sana_sprint":
        from dit_accel.pipeline import reset_caches
        reset_caches(pipe)
    else:
        from dit_accel.pipeline_dit import reset_dit_caches
        reset_dit_caches(pipe)


def _run(pipe, model_name: str, items, num_steps: int, generators):
    if model_name == "sana_sprint":
        if num_steps != 2:
            return pipe(
                items, num_inference_steps=num_steps,
                generator=generators, intermediate_timesteps=None,
            ).images
        return pipe(items, num_inference_steps=num_steps, generator=generators).images
    return pipe(
        class_labels=items, num_inference_steps=num_steps, generator=generators,
    ).images


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="generate")
def main(cfg: DictConfig) -> None:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        print(OmegaConf.to_yaml(cfg))

    schedule = _load_schedule(cfg, cfg.variant.name, rank)
    pipe = _build_pipe(cfg, device, schedule, rank)

    all_items = _conditioning(cfg)
    if cfg.get("limit") is not None:
        all_items = all_items[: int(cfg.limit)]
    total = len(all_items)
    per_rank = (total + world_size - 1) // world_size
    rank_start = rank * per_rank
    rank_end = min(rank_start + per_rank, total)
    my_items = all_items[rank_start:rank_end]
    if rank == 0:
        print(f"World size {world_size}, total {total}, per-rank {per_rank}")

    base_seed = cfg.seed + rank * 1_000_000
    timer = LatencyTimer()
    reset_peak_memory()

    for warm_item in my_items[:2]:
        with timer.warmup():
            _run(pipe, cfg.model.name, [warm_item], cfg.num_steps,
                 [torch.Generator(device).manual_seed(0)])
        _reset(pipe, cfg.model.name)

    t0 = time.time()
    written = 0
    for batch_start in range(0, len(my_items), cfg.batch_size):
        batch = my_items[batch_start:batch_start + cfg.batch_size]
        batch_seeds = [base_seed + rank_start + batch_start + i for i in range(len(batch))]
        generators = [torch.Generator(device).manual_seed(s) for s in batch_seeds]

        _reset(pipe, cfg.model.name)

        with timer.measure():
            images = _run(pipe, cfg.model.name, batch, cfg.num_steps, generators)

        for i, img in enumerate(images):
            global_idx = rank_start + batch_start + i
            final_path = output / f"{global_idx:08d}.png"
            tmp_path = output / f"{global_idx:08d}.png.tmp"
            img.save(tmp_path, format="PNG", optimize=False)
            os.replace(tmp_path, final_path)
            written += 1

        if rank == 0 and (batch_start // cfg.batch_size) % 50 == 0:
            elapsed = time.time() - t0
            done = batch_start + len(batch)
            eta = elapsed * (len(my_items) - done) / max(done, 1)
            print(f"  rank 0: {done}/{len(my_items)} in {elapsed:.0f}s (eta {eta:.0f}s)")

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        summary = timer.summary()
        peak_mem = peak_memory_gb()
        median_ms = summary.get("median_ms") or 0.0
        images_per_sec = (cfg.batch_size * 1000.0 / median_ms) if median_ms > 0 else 0.0

        cache_stats: dict = {}
        if hasattr(pipe, "_dit_accel_block_cache"):
            cache_stats["block_cache"] = pipe._dit_accel_block_cache.stats()
        if hasattr(pipe, "_dit_accel_xattn_stats"):
            xs = pipe._dit_accel_xattn_stats
            total_x = xs.hits + xs.misses
            cache_stats["xattn_cache"] = {
                "hits": xs.hits, "misses": xs.misses, "total": total_x,
                "hit_rate": (xs.hits / total_x) if total_x else 0.0,
            }
        if hasattr(pipe, "_dit_accel_state_cache"):
            cache_stats["lacache"] = pipe._dit_accel_state_cache.stats()
        if hasattr(pipe, "_dit_accel_sparsity"):
            cache_stats["sparse_ffn"] = pipe._dit_accel_sparsity.stats()

        manifest = {
            "variant": cfg.variant.name,
            "model": cfg.model.name,
            "num_steps": cfg.num_steps,
            "samples_per_class": cfg.samples_per_class,
            "batch_size": cfg.batch_size,
            "world_size": world_size,
            "k_per_step": cfg.k_per_step,
            "dataset": cfg.dataset.name,
            "schedule_path": str(cfg.schedule_path) if cfg.schedule_path else None,
            "latency_ms": summary,
            "images_per_sec_rank0": images_per_sec,
            "peak_mem_gb_rank0": peak_mem,
            "cache_stats_rank0": cache_stats,
            "total_images": total,
            "wall_time_s": time.time() - t0,
            "sana_ffn_sparse": (
                pipe._dit_accel_sana_ffn_sparse.summary()
                if hasattr(pipe, "_dit_accel_sana_ffn_sparse") else None
            ),
        }
        with open(output / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Wrote {written} images on rank 0; manifest -> {output / 'manifest.json'}")
        print(json.dumps(manifest, indent=2))

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
