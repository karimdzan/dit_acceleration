# FAE-Adapters: Feature Auto-Encoders for Frozen Vision Encoders + Pretrained DiTs and Flow Matching Models

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
pip install diffusers>=0.37.0 transformers accelerate safetensors peft
```

## Guide

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