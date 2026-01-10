import argparse
from pathlib import Path
import json

import torch

from pfd_dit.backbones.sd15 import SD15Backbone
from pfd_dit.train_sd15 import TrainSD15Config, finetune_sd15_unet
from pfd_dit.eval_sd15 import EvalSD15Config, evaluate_sd15_teacher_student


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)

    ap.add_argument("--teacher_id", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--student_id", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5")

    ap.add_argument("--train_size", type=int, required=True)
    ap.add_argument("--coco_split", type=str, default="train")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16"])  # V100-safe

    ap.add_argument("--train_steps", type=int, default=600)
    ap.add_argument("--train_bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-6)

    ap.add_argument("--num_inference_steps", type=int, default=30)
    ap.add_argument("--guidance_scale", type=float, default=7.5)

    ap.add_argument("--eval_samples", type=int, default=128)
    ap.add_argument("--eval_bs", type=int, default=4)
    ap.add_argument("--descriptor", type=str, default="clip")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher_dir = run_dir / "teacher_data"
    from scripts.generate_teacher_dataset_sd15 import main as gen_main
    import subprocess, sys
    subprocess.check_call([
        sys.executable, "-m", "scripts.generate_teacher_dataset_sd15",
        "--teacher_id", args.teacher_id,
        "--out_dir", str(teacher_dir),
        "--train_size", str(args.train_size),
        "--coco_split", args.coco_split,
        "--seed", str(args.seed),
        "--dtype", args.dtype,
        "--num_inference_steps", str(args.num_inference_steps),
        "--guidance_scale", str(args.guidance_scale),
        "--batch_size", str(args.train_bs),
    ])

    student_out = run_dir / "student"
    tcfg = TrainSD15Config(
        train_dir=teacher_dir,
        out_dir=student_out,
        max_steps=args.train_steps,
        batch_size=args.train_bs,
        lr=args.lr,
        seed=args.seed,
        mixed_precision=args.mixed_precision,
        grad_clip=1.0,
        gradient_checkpointing=True,
    )
    student_unet_dir = finetune_sd15_unet(
        student_id=args.student_id,
        cfg=tcfg,
        device=device,
        torch_dtype=torch_dtype,
    )

    teacher = SD15Backbone.from_pretrained(args.teacher_id, device=device, torch_dtype=torch_dtype)
    student = SD15Backbone.from_pretrained(args.student_id, device=device, torch_dtype=torch_dtype)

    ecfg = EvalSD15Config(
        out_path=run_dir / "metrics.json",
        eval_samples=args.eval_samples,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed + 123,
        descriptor=args.descriptor,
        batch_size=args.eval_bs,
        fid_batch_size=max(4, min(8, args.eval_bs)),
        empirical_chunk_train=1024,
    )

    out = evaluate_sd15_teacher_student(
        teacher=teacher,
        student=student,
        train_dir=teacher_dir,
        student_unet_dir=student_unet_dir,
        cfg=ecfg,
    )

    (run_dir / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Ok: metrics: {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
