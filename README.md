# Accelerating DiT | PFD / Memorization vs Generalization for a pretrained DiT

This repo implements the **teacher–student evaluation protocol** and **Probability Flow Distance (PFD)** idea from:

- Zhang et al., *Understanding Generalization in Diffusion Models via Probability Flow Distance* (arXiv:2505.20123).

We use a **pretrained Diffusion Transformer (DiT)** from diffusers as the **teacher** and create a **student** by fine-tuning (small, controllable training) on a limited synthetic dataset sampled from the teacher.

## What you get
- `PFD(teacher, student)`: proxy **generalization error** under the paper’s teacher–student protocol.
- `M-distance(student, trainset)`: a practical **memorization** proxy (matches the paper’s discussion of memorization metrics).

> Note: The paper also derives an *exact* memorization score via the closed-form score of the empirical distribution (Appendix D). That requires implementing an explicit PF-ODE solver for the empirical distribution; this repo keeps the memorization metric practical and scalable.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .

# 1) Generate a small synthetic trainset from the teacher
python -m scripts.generate_teacher_dataset \
  --out_dir runs/demo \
  --teacher_id facebook/DiT-XL-2-256 \
  --train_size 64 \
  --image_size 256 \
  --num_inference_steps 30

# 2) Fine-tune student on that set (few steps) and evaluate
python -m scripts.run_experiment \
  --run_dir runs/demo \
  --teacher_id facebook/DiT-XL-2-256 \
  --train_size 64 \
  --max_train_steps 600 \
  --lr 5e-5 \
  --eval_samples 256 \
  --num_inference_steps 30 \
  --descriptor clip
```

## Tips for seeing memorization vs generalization
- Memorization regime: small `--train_size` (e.g., 16/32/64) + longer training (more steps) + slightly higher LR.
- Generalization regime: larger `--train_size` (e.g., 5k+) + moderate steps.

If you have limited compute, sweep sizes like: 32, 64, 128, 256, 512.

## Outputs
In `run_dir/`:
- `teacher_train_latents.pt` : synthetic dataset (latents + labels)
- `student/` : fine-tuned student pipeline
- `metrics.json` : evaluation results (PFD + M-distance + metadata)
