"""Calibrate one of: block_cache, linear_attn_state, sparsity, sana_ffn_groups."""
import json
import random
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from ._common import CONFIG_PATH


def _calibration_prompts(cfg: DictConfig, num_prompts: int) -> list:
    if cfg.dataset.name == "coco30k":
        from dit_accel.data import load_coco_30k_captions
        return random.sample(load_coco_30k_captions(cfg.dataset.coco_local_dir), num_prompts)
    if cfg.model.name == "sana_sprint":
        from dit_accel.evaluation.imagenet_prompts import PROMPT_TEMPLATE, load_class_names
        names = load_class_names()
        sampled = random.sample(names, num_prompts)
        return [PROMPT_TEMPLATE.format(label=n) for n in sampled]
    return [random.randint(0, 999) for _ in range(num_prompts)]


def _run_pipe(pipe, model_name: str, item, num_steps: int, gen=None):
    if model_name == "sana_sprint":
        if num_steps != 2:
            return pipe(item, num_inference_steps=num_steps, generator=gen, intermediate_timesteps=None).images
        return pipe(item, num_inference_steps=num_steps, generator=gen).images
    return pipe(class_labels=[item], num_inference_steps=num_steps, generator=gen).images


def _calibrate_block_cache(cfg: DictConfig) -> None:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.model.name == "sana_sprint":
        from dit_accel.pipeline import load_pipeline
        pipe = load_pipeline(variant="block_cache", checkpoint=cfg.model.checkpoint)
    else:
        from dit_accel.pipeline_dit import load_dit_pipeline
        pipe = load_dit_pipeline(variant="block_cache", checkpoint=cfg.model.checkpoint)

    store = pipe._dit_accel_block_cache
    store.calibration_mode = True

    prompts = _calibration_prompts(cfg, cfg.num_prompts)
    agg: dict[tuple[int, int], list[float]] = {}

    for i, item in enumerate(prompts):
        store.clear()
        store.calibration_mode = True
        gen = torch.Generator(pipe.device).manual_seed(cfg.seed + i)
        with torch.no_grad():
            _run_pipe(pipe, cfg.model.name, item, cfg.num_steps, gen)
        for key, val in store.deltas.items():
            agg.setdefault(key, []).append(val)
        if (i + 1) % 16 == 0:
            print(f"  calibrated {i + 1}/{len(prompts)}")

    summary: dict[int, dict[int, dict]] = {}
    for (block_idx, step_idx), vals in agg.items():
        summary.setdefault(int(step_idx), {})[int(block_idx)] = {
            "mean": float(sum(vals) / len(vals)),
            "n": len(vals),
            "min": float(min(vals)),
            "max": float(max(vals)),
        }

    payload = {
        "model": cfg.model.name,
        "num_prompts": cfg.num_prompts,
        "num_steps": cfg.num_steps,
        "deltas_by_step": summary,
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    with open(output.with_suffix(".json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {output}")


def _calibrate_linear_attn_state(cfg: DictConfig) -> None:
    from dit_accel.evaluation.imagenet_prompts import PROMPT_TEMPLATE, load_class_names
    from dit_accel.pipeline import load_pipeline, reset_caches

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    pipe = load_pipeline(variant="lacache", checkpoint=cfg.model.checkpoint)
    store = pipe._dit_accel_state_cache
    store.calibration_mode = True

    names = load_class_names()
    sampled = random.sample(names, cfg.num_prompts)
    prompts = [PROMPT_TEMPLATE.format(label=n) for n in sampled]

    agg: dict[tuple[int, int], list[float]] = {}
    for i, prompt in enumerate(prompts):
        reset_caches(pipe)
        store.calibration_mode = True
        store.deltas.clear()
        _ = pipe(prompt, num_inference_steps=cfg.num_steps).images
        for key, val in store.deltas.items():
            agg.setdefault(key, []).append(val)
        if (i + 1) % 16 == 0:
            print(f"  calibrated {i + 1}/{len(prompts)} prompts")

    summary = {}
    for key, vals in agg.items():
        layer, step = key
        summary.setdefault(step, {})[layer] = {
            "mean": float(sum(vals) / len(vals)),
            "n": len(vals),
            "min": min(vals),
            "max": max(vals),
        }

    payload = {
        "num_prompts": cfg.num_prompts,
        "num_steps": cfg.num_steps,
        "deltas_by_step": summary,
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    with open(output.with_suffix(".json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved calibration deltas to {output}")


def _calibrate_sparsity(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg.model.name == "sana_sprint":
        from dit_accel.evaluation.imagenet_prompts import build_prompts
        from dit_accel.pipeline import load_pipeline, reset_caches
        from dit_accel.sparsity.mixffn_sparsifier import install_mixffn_sparsifier
        from dit_accel.sparsity.store import fit_thresholds_from_samples
        pipe = load_pipeline(variant="bf16", checkpoint=cfg.model.checkpoint, device=device)
        store = install_mixffn_sparsifier(pipe, thresholds=None, calibration_mode=True)
        store.calibration_samples_per_batch = cfg.samples_per_batch
        prompts = build_prompts(samples_per_class=1)

        t0 = time.time()
        for b in range(cfg.num_batches):
            start = (b * cfg.batch_size) % len(prompts)
            batch = prompts[start:start + cfg.batch_size]
            if len(batch) < cfg.batch_size:
                batch = batch + prompts[: cfg.batch_size - len(batch)]
            with torch.no_grad():
                if cfg.num_steps != 2:
                    _ = pipe(
                        prompt=batch, num_inference_steps=cfg.num_steps,
                        guidance_scale=4.5, intermediate_timesteps=None,
                        generator=torch.Generator(device=device).manual_seed(cfg.seed + b),
                    )
                else:
                    _ = pipe(
                        prompt=batch, num_inference_steps=cfg.num_steps,
                        guidance_scale=4.5,
                        generator=torch.Generator(device=device).manual_seed(cfg.seed + b),
                    )
            reset_caches(pipe)
            if (b + 1) % 4 == 0 or b == 0:
                print(f"  batch {b + 1}/{cfg.num_batches}  elapsed {time.time() - t0:.1f}s")
        model_tag = "sana_sprint"
    else:
        from dit_accel.pipeline_dit import load_dit_pipeline, reset_dit_caches
        from dit_accel.sparsity.dit_mlp_sparsifier import install_dit_mlp_sparsifier
        from dit_accel.sparsity.store import fit_thresholds_from_samples
        pipe = load_dit_pipeline(variant="bf16", checkpoint=cfg.model.checkpoint, device=device)
        store = install_dit_mlp_sparsifier(pipe, thresholds=None, calibration_mode=True)
        store.calibration_samples_per_batch = cfg.samples_per_batch

        t0 = time.time()
        g = torch.Generator(device=device)
        num_classes = 1000
        for b in range(cfg.num_batches):
            start = (b * cfg.batch_size) % num_classes
            ids = [(start + i) % num_classes for i in range(cfg.batch_size)]
            class_ids = torch.tensor(ids, dtype=torch.long, device=device)
            g.manual_seed(cfg.seed + b)
            with torch.no_grad():
                _ = pipe(class_labels=class_ids, num_inference_steps=cfg.num_steps, generator=g)
            reset_dit_caches(pipe)
            if (b + 1) % 4 == 0 or b == 0:
                print(f"  batch {b + 1}/{cfg.num_batches}  elapsed {time.time() - t0:.1f}s")
        model_tag = "dit_xl"

    n_layers = len(store.calibration_samples)
    if n_layers == 0:
        raise SystemExit("Calibration recorded zero layers.")
    sample_count = sum(
        sum(t.numel() for t in chunks)
        for chunks in store.calibration_samples.values()
    ) // n_layers
    print(f"Calibrated {n_layers} layers (~{sample_count} samples each).")
    print(f"Fitting per-layer thresholds at q={cfg.target_sparsity}...")
    thresholds = fit_thresholds_from_samples(
        store.calibration_samples, target_sparsity=cfg.target_sparsity,
    )
    for k in sorted(thresholds):
        print(f"  {k}: {float(thresholds[k]):.4e}")

    payload = {
        "thresholds": {k: v.cpu() for k, v in thresholds.items()},
        "target_sparsity": cfg.target_sparsity,
        "model": model_tag,
        "num_steps": cfg.num_steps,
        "num_calib_batches": cfg.num_batches,
        "per_layer_samples": sample_count,
    }
    output = Path(cfg.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(f"Saved sparsity calibration -> {output}")


def _calibrate_sana_ffn_groups(cfg: DictConfig) -> None:
    from tqdm.auto import tqdm

    from dit_accel.evaluation.imagenet_prompts import build_prompts
    from dit_accel.pipeline import load_pipeline, reset_caches
    from dit_accel.sparsity.sana_sparse_ffn import install_sana_ffn_group_observer, save_group_plan

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_device(0)

    prompts = build_prompts(samples_per_class=1)[: cfg.num_prompts]
    pipe = load_pipeline(variant="bf16", checkpoint=cfg.model.checkpoint, device=device)
    observer = install_sana_ffn_group_observer(pipe, group_size=cfg.group_size)

    t0 = time.time()
    for batch_start in tqdm(range(0, len(prompts), cfg.batch_size), desc="calibrate Sana FFN groups"):
        batch = prompts[batch_start: batch_start + cfg.batch_size]
        generators = [torch.Generator(device).manual_seed(cfg.seed + batch_start + i) for i in range(len(batch))]
        reset_caches(pipe)
        if cfg.num_steps != 2:
            _ = pipe(batch, num_inference_steps=cfg.num_steps, generator=generators, intermediate_timesteps=None).images
        else:
            _ = pipe(batch, num_inference_steps=cfg.num_steps, generator=generators).images

    plan = observer.build_plan(cfg.keep_ratio)
    plan["calibration"] = {
        "num_prompts": cfg.num_prompts,
        "num_steps": cfg.num_steps,
        "batch_size": cfg.batch_size,
        "seed": cfg.seed,
        "wall_time_s": time.time() - t0,
    }
    save_group_plan(plan, cfg.output)

    summary = {
        "output": str(cfg.output),
        "keep_ratio": cfg.keep_ratio,
        "group_size": cfg.group_size,
        "num_layers": len(plan["layers"]),
        "groups_per_layer": {
            str(k): {
                "num_groups": int(v["num_groups"]),
                "keep_groups": int(v["keep_groups"]),
                "hidden_channels": int(v["hidden_channels"]),
            }
            for k, v in plan["layers"].items()
        },
    }
    print(json.dumps(summary, indent=2))


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="calibrate")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    mode = cfg.mode
    if mode == "block_cache":
        _calibrate_block_cache(cfg)
    elif mode == "linear_attn_state":
        _calibrate_linear_attn_state(cfg)
    elif mode == "sparsity":
        _calibrate_sparsity(cfg)
    elif mode == "sana_ffn_groups":
        _calibrate_sana_ffn_groups(cfg)
    else:
        raise SystemExit(f"unknown calibrate mode: {mode}")


if __name__ == "__main__":
    main()
