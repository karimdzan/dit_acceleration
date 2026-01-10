import argparse, json
from pathlib import Path

import torch
from tqdm import tqdm

from pfd_dit.backbones.sd15 import SD15Backbone
from pfd_dit.data import sample_prompts_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_id", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--train_size", type=int, required=True)

    ap.add_argument("--coco_split", type=str, default="train")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--num_inference_steps", type=int, default=30)
    ap.add_argument("--guidance_scale", type=float, default=7.5)
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    prompts = sample_prompts_list(num_prompts=args.train_size, split=args.coco_split, seed=args.seed)
    (out_dir / "prompts.json").write_text(json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8")

    teacher = SD15Backbone.from_pretrained(args.teacher_id, device=device, torch_dtype=torch_dtype)

    cond_all, uncond_all = [], []
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Encode prompts"):
        p = prompts[i:i+args.batch_size]
        cond, uncond = teacher.encode_prompts(p)
        cond_all.append(cond.detach().cpu())
        uncond_all.append(uncond.detach().cpu())
    cond_all = torch.cat(cond_all, dim=0)
    uncond_all = torch.cat(uncond_all, dim=0)
    torch.save({"cond": cond_all, "uncond": uncond_all}, out_dir / "prompt_embeds.pt")

    prompt_ids = torch.arange(len(prompts), dtype=torch.long)
    torch.save({"prompt_ids": prompt_ids}, out_dir / "prompt_ids.pt")

    latents_out = []
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Sample teacher x0 latents"):
        cond = cond_all[i:i+args.batch_size].to(device=device, dtype=torch_dtype)
        uncond = uncond_all[i:i+args.batch_size].to(device=device, dtype=torch_dtype)

        x0_lat, _ = teacher.sample_latents_and_images(
            cond_uncond_embeds=(cond, uncond),
            init_latents=None,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        )
        latents_out.append(x0_lat.detach().cpu())

    latents_out = torch.cat(latents_out, dim=0)
    torch.save({"latents": latents_out}, out_dir / "train_latents.pt")

    print(f"Ok: wrote teacher dataset to: {out_dir}")
    print("  prompts.json, prompt_embeds.pt, prompt_ids.pt, train_latents.pt")


if __name__ == "__main__":
    main()
