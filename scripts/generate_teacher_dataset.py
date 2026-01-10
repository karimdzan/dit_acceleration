import argparse
from pathlib import Path
import torch

from pfd_dit.models import load_teacher_student
from pfd_dit.teacher_dataset import TeacherDatasetConfig, generate_teacher_latents
from pfd_dit.utils.device import pick_device

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--teacher_id", type=str, default="facebook/DiT-XL-2-256")
    ap.add_argument("--train_size", type=int, default=64)
    ap.add_argument("--num_inference_steps", type=int, default=30)
    ap.add_argument("--guidance_scale", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    args = ap.parse_args()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    device = pick_device()

    run_dir = Path(args.out_dir)
    out_path = run_dir / "teacher_train_latents.pt"

    pipes = load_teacher_student(args.teacher_id, torch_dtype=dtype, device=device, student_from_teacher=True)
    cfg = TeacherDatasetConfig(
        out_path=out_path,
        train_size=args.train_size,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    generate_teacher_latents(pipes.teacher, cfg, device=device)
    print(f"Ok: saved: {out_path}")

if __name__ == "__main__":
    main()
