# DiT Runtime Acceleration

Training-free inference acceleration for diffusion transformers. Covers
Sana Sprint (linear-attention, 1-4 steps) and DiT-XL (softmax, 25 steps)
on ImageNet-1K and MS-COCO 30K with FID and CLIP evaluation.

The repo is exposes four console entry points: `dit-accel-generate`, `dit-accel-calibrate`,
`dit-accel-evaluate`, and `dit-accel-prepare`.

## Install

```bash
pip install -e .
# optional: Triton kernel for column-sparse conv_point
pip install -e ".[triton]"
```

Set checkpoint and data paths via environment variables (the configs
read them with sensible defaults):

```bash
export SANA_SPRINT_PATH=/path/to/sana_sprint_1.6b
export DIT_XL_PATH=/path/to/dit_xl_256
export COCO_LOCAL_DIR=data/coco_val2014_30k
export IMAGENET_VAL_DIR=/path/to/imagenet_val
```

## Hydra configs

```
configs/
  generate.yaml       generate samples for a variant
  calibrate.yaml      calibrate a cache or sparsity schedule
  evaluate.yaml       compute FID + CLIP for a samples dir
  prepare.yaml        prepare COCO-30K / ImageNet val / FID stats
  model/{sana_sprint,dit_xl}.yaml
  dataset/{imagenet,coco30k}.yaml
  variant/{bf16,int8,int8_bnb,xattn,lacache,cached,block_cache,sparse_ffn,gsparse}.yaml
  sana_ffn/{static,dynamic}.yaml
```

Each entry point picks its config root (`generate.yaml` etc.) and
accepts Hydra overrides on the CLI.

## Quick start

Prepare references:

```bash
dit-accel-prepare task=coco30k local_dir=data/coco_val2014_30k \
    register_fid=coco_val2014_30k
dit-accel-prepare task=imagenet_val src_dir=$IMAGENET_VAL_DIR \
    local_dir=data/imagenet_val_50k samples_per_class=50 \
    register_fid=imagenet_val_50k
```

Generate (DDP-friendly: launch with `torchrun --nproc_per_node=N`):

```bash
torchrun --nproc_per_node=4 -m dit_accel.cli.generate \
    model=sana_sprint variant=bf16 dataset=imagenet \
    num_steps=4 samples_per_class=50 batch_size=4 \
    output_dir=samples/sana_bf16_4step
```

Calibrate a block-cache schedule:

```bash
dit-accel-calibrate mode=block_cache model=sana_sprint dataset=imagenet \
    num_steps=4 num_prompts=256 \
    output=calibration/block_sana_imagenet_4step.pt
```

Calibrate sparsity thresholds:

```bash
dit-accel-calibrate mode=sparsity model=dit_xl \
    num_steps=25 target_sparsity=0.5 \
    output=calibration/sparsity_dit_xl_25step.pt
```

Calibrate static Sana FFN group plan:

```bash
dit-accel-calibrate mode=sana_ffn_groups model=sana_sprint \
    num_prompts=256 num_steps=4 keep_ratio=0.90 group_size=32 \
    output=calibration/sana_ffn_groups_keep90.pt
```

Generate with the block cache and calibrated schedule:

```bash
torchrun --nproc_per_node=4 -m dit_accel.cli.generate \
    model=sana_sprint variant=block_cache dataset=imagenet \
    num_steps=4 samples_per_class=50 batch_size=4 \
    schedule_path=calibration/block_sana_imagenet_4step.pt \
    k_per_step=4 \
    output_dir=samples/sana_block_k4_4step
```

Combine variants with `+` (Hydra-friendly: still a single value):

```bash
torchrun --nproc_per_node=4 -m dit_accel.cli.generate \
    model=sana_sprint variant=block_cache dataset=imagenet \
    +variant.name=int8+xattn+block_cache num_steps=4 ...
```

Evaluate:

```bash
dit-accel-evaluate dataset=imagenet \
    samples_dir=samples/sana_block_k4_4step \
    ref_name=imagenet_val_50k \
    output=results/sana_block_k4_4step.json
```

## Variants

| variant          | applies to       | effect |
|------------------|------------------|--------|
| `bf16`           | sana / dit       | baseline |
| `int8`           | sana / dit       | torchao weight-only int8 |
| `int8_bnb`       | sana / dit       | bitsandbytes 8-bit (legacy) |
| `xattn`          | sana             | cross-attn KV cache (exact) |
| `lacache`        | sana             | linear-attention state cache |
| `cached`         | sana             | `xattn + lacache` |
| `block_cache`    | sana / dit       | block-residual feature cache |
| `sparse_ffn`     | sana / dit       | CATS/TEAL activation sparsifier |
| `gsparse`        | sana             | group-sparse FFN compact convs |

Combine via `+` in the variant name (`int8+xattn+block_cache`).

## Layout

```
src/dit_accel/
  pipeline.py              Sana Sprint factory
  pipeline_dit.py          DiT-XL factory
  quant_compat.py          torchao version shim
  caching/                 block / cross-attn / linear-attn caches
  sparsity/                Mix-FFN, DiT MLP, Sana group-sparse FFN, Triton kernel
  evaluation/              FID, CLIP, latency, ImageNet prompts
  data/                    COCO-30K loader, ImageNet val builder
  cli/                     Hydra entry points
configs/                   Hydra config tree
tests/                     pytest-compatible unit tests
```
