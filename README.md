# FAE-Adapters: Flexible Feature Auto-Encoders for Frozen Vision Encoders + Pretrained DiTs

This repository implements the core method from **"One Layer Is Enough: Adapting Pretrained Visual Encoders for Image Generation"** and rewrites the generator stage around a **backend registry** so you can plug FAE latents into multiple diffusion-transformer families.

It keeps the paper's main tokenizer design:
- a **frozen visual encoder** (DINOv2, SigLIP2, ViT-MAE)
- a **single-attention latent encoder**
- a **6-layer feature decoder**
- a **ViT-style pixel decoder**
- a generator stage trained or finetuned **directly on the compact FAE latents**

The new part is the generator abstraction:
- **internal** latent DiT for smoke tests and full local training
- **diffusers/DiT** backend for class-conditional latent diffusion transformers
- **diffusers/SD3** backend for MMDiT-style text-conditioned models
- **diffusers/Sana / Sana-Sprint** backend for small linear-DiT style text-conditioned models
- a **bridge module** that maps between FAE token latents and the latent tensor format expected by the target generator

## Why this rewrite

The original minimal repo used a single in-repo DiT with token-shaped latents. Pretrained diffusion transformers do **not** share the same latent shape, conditioning interface, or denoising objective.

This rewrite makes stage 3 modular:

```text
images
  -> frozen vision encoder
  -> single-attention FAE encoder
  -> compact FAE tokens [B, N, d]
  -> token/grid bridge
  -> backend-specific latents [B, C, H, W]
  -> pretrained DiT / MMDiT / Sana transformer
  -> token/grid bridge (inverse)
  -> FAE feature decoder
  -> pixel decoder
```

## What is implemented

### Stage 1: feature auto-encoder
- frozen vision encoders:
  - `dinov2`
  - `siglip2`
  - `vit_mae`
- single self-attention latent encoder
- 6-layer transformer feature decoder
- diagonal Gaussian posterior + KL regularization

### Stage 2: pixel decoder
- ViT-style pixel decoder
- Gaussian embedding decoder pretraining
- fine-tuning on reconstructed FAE features
- optional GAN discriminator/perceptual loss

### Stage 3: generator backends
- `internal_dit`: simple in-repo latent DiT over image-like latent tensors
- `diffusers_dit`: wrapper for `diffusers.DiTTransformer2DModel`
- `diffusers_sd3`: wrapper for `diffusers.SD3Transformer2DModel`
- `diffusers_sana`: wrapper for `diffusers.SanaTransformer2DModel`

### Conditioning modes
- class labels
- pooled text embeddings for the internal backend
- native prompt encoding through a diffusers pipeline for SD3 / Sana / Sana-Sprint

## Key design choices

### 1) Bridge instead of hard-coded latent format
FAE produces token latents `[B, N, d]`. Most pretrained diffusion transformers expect image-like latent tensors `[B, C, H, W]` with model-specific `C`, `H`, `W`, and sometimes fixed positional embeddings. The bridge converts between these spaces with:
- token/grid reshaping
- spatial resizing
- learnable channel projection
- optional residual conv blocks

### 2) Backend contracts instead of one-off patches
Every generator backend exposes the same interface:
- `training_loss(latents, conditioning)`
- `sample_latents(batch_size, conditioning)`
- `latent_spec()`
- `encode_prompts(prompts)` when the backend owns the prompt encoder

Adding a new backend means implementing a small subclass instead of rewriting the whole training pipeline.

### 3) Objective adapters
Different transformer families use different denoising objectives. The code supports:
- standard diffusion (`epsilon`, `v_prediction`, `sample`)
- flow-matching style training
- distilled / consistency-style inference hooks for already-distilled checkpoints

## Supported backends and intended use

| Backend | Use case | Notes |
|---|---|---|
| `internal_dit` | local debugging, unit tests, small-scale experiments | no external checkpoint required |
| `diffusers_dit` | class-conditional pretrained DiT | best first target for testing FAE + pretrained DiT |
| `diffusers_sd3` | text-conditioned MMDiT | requires SD3 pipeline assets and native prompt encoding |
| `diffusers_sana` | text-conditioned Sana or Sana-Sprint | good small-model target on H100; supports BF16 pipelines |

## H100-oriented notes

The repo is structured so you can test several strong small or medium transformer backbones on a single H100:
- BF16 everywhere by default when available
- gradient checkpointing toggle
- optional `torch.compile`
- optional `channels_last`
- LoRA / frozen / full-finetune modes for the external backend
- bridge-only finetuning for quick adaptation checks

Suggested first experiments:
1. train stage 1 and stage 2 with DINOv2 or SigLIP2
2. freeze backbone + FAE + pixel decoder
3. attach `diffusers_dit` or `diffusers_sana`
4. train only the bridge for a short sanity run
5. enable LoRA on the transformer if the bridge alone is not enough

## Repository layout

```text
fae/
  backbones/          frozen vision encoders
  models/             FAE, pixel decoder, posterior, bridge
  generators/         backend registry + diffusion wrappers
  training/           stage 1/2/3 loops
  scripts/            train / sample entrypoints
  utils/              checkpointing, image utils, loss helpers
configs/
  encoder/
  train/
tests/
```

## Install

```bash
pip install -e .
```

Optional but recommended for external backends:

```bash
pip install diffusers>=0.37.0 transformers accelerate safetensors peft
```

## Quickstart

### Stage 1
```bash
python -m fae.scripts.train_stage1 --config configs/train/stage1_dinov2.yaml
```

### Stage 2
```bash
python -m fae.scripts.train_stage2 --config configs/train/stage2_pixel_gaussian.yaml
python -m fae.scripts.train_stage2 --config configs/train/stage2_pixel_finetune.yaml
```

### Stage 3 with internal backend
```bash
python -m fae.scripts.train_stage3 --config configs/train/stage3_internal_class.yaml
```

### Stage 3 with Sana-Sprint bridge finetuning
```bash
python -m fae.scripts.train_stage3 --config configs/train/stage3_sana_sprint_0p6b.yaml
```

### Sampling through the FAE decoder stack
```bash
python -m fae.scripts.sample --config configs/train/stage3_sana_sprint_0p6b.yaml --prompt "a tiny astronaut hatching from an egg on the moon"
```

## Important caveat

This repo is designed to make FAE-style latent adaptation **easy to test on pretrained DiT-family backbones**. It does **not** claim to fully reproduce every original backend training recipe. In particular:
- SD3 native training uses a full MMDiT text stack
- Sana-Sprint is a distilled few-step model with its own consistency/distillation recipe
- this repo lets you **reuse those pretrained transformers as backends**, finetune bridges, and optionally add LoRA / partial finetuning

That is the intended experimental interface for testing the paper's method on modern pretrained diffusion transformers.
