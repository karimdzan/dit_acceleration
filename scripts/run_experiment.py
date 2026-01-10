import argparse
from pathlib import Path
import torch

from pfd_dit.data import load_latents
from pfd_dit.models import load_teacher_student
from pfd_dit.train import TrainConfig, finetune_student_on_latents
from pfd_dit.eval import EvalConfig, evaluate_teacher_student
from pfd_dit.teacher_dataset import TeacherDatasetConfig, generate_teacher_latents
from pfd_dit.utils.device import pick_device

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--teacher_id", type=str, default="facebook/DiT-XL-2-256")

    ap.add_argument("--train_size", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--num_inference_steps", type=int, default=30)
    ap.add_argument("--guidance_scale", type=float, default=4.0)

    ap.add_argument("--max_train_steps", type=int, default=600)
    ap.add_argument("--train_batch_size", type=int, default=8)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--eval_samples", type=int, default=256)
    ap.add_argument("--eval_seed", type=int, default=123)
    ap.add_argument("--descriptor", type=str, default="clip", choices=["identity", "clip"])
    ap.add_argument("--eval_batch_size", type=int, default=8)

    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    args = ap.parse_args()

    if args.mixed_precision == "no":
        dtype = torch.float32
    else:
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    device = pick_device()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    train_latents_path = run_dir / "teacher_train_latents.pt"

    pipes = load_teacher_student(args.teacher_id, torch_dtype=dtype, device=device, student_from_teacher=True)

    if not train_latents_path.exists():
        cfg_data = TeacherDatasetConfig(
            out_path=train_latents_path,
            train_size=args.train_size,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )
        generate_teacher_latents(pipes.teacher, cfg_data, device=device)

    train_latents, train_labels = load_latents(train_latents_path)

    cfg_train = TrainConfig(
        out_dir=run_dir,
        max_train_steps=args.max_train_steps,
        train_batch_size=args.train_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        mixed_precision=args.mixed_precision,
        seed=args.seed,
    )
    student_dir = finetune_student_on_latents(
        student_pipe=pipes.student,
        train_latents=train_latents,
        train_labels=train_labels,
        cfg=cfg_train,
    )
    print(f"ok: student saved: {student_dir}")

    from diffusers import DiTPipeline
    pipes.student = DiTPipeline.from_pretrained(student_dir, torch_dtype=dtype).to(device)
    pipes.student.scheduler = pipes.teacher.scheduler

    metrics_path = run_dir / "metrics.json"
    cfg_eval = EvalConfig(
        out_path=metrics_path,
        eval_samples=args.eval_samples,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.eval_seed,
        descriptor=args.descriptor,
        batch_size=args.eval_batch_size,
    )

    out = evaluate_teacher_student(
        teacher_pipe=pipes.teacher,
        student_pipe=pipes.student,
        train_latents=train_latents,
        train_labels=train_labels,
        cfg=cfg_eval,
    )

    print("RESULT:")
    for k, v in out.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
