from dataclasses import dataclass
from pathlib import Path
import json

import torch
from tqdm import tqdm
from diffusers import DDIMScheduler, UNet2DConditionModel


from pfd_dit.backbones.sd15 import SD15Backbone
from pfd_dit.metrics.descriptors import build_descriptor, DescriptorName
from pfd_dit.metrics.pfd import estimate_pfd, m_distance_to_trainset
from pfd_dit.metrics.fid import compute_fid
from pfd_dit.metrics.empirical_pfd_cond import prepare_empirical_state, sample_empirical_ddim_latents


@dataclass
class EvalSD15Config:
    out_path: Path
    eval_samples: int = 128
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    seed: int = 123
    descriptor: DescriptorName = "clip"
    batch_size: int = 4

    fid_batch_size: int = 8
    empirical_chunk_train: int = 1024


@torch.no_grad()
def evaluate_sd15_teacher_student(
    *,
    teacher: SD15Backbone,
    student: SD15Backbone,
    train_dir: Path,
    student_unet_dir: Path,
    cfg: EvalSD15Config,
) -> dict:
    torch.manual_seed(cfg.seed)
    device = teacher.device
    bs = int(cfg.batch_size)

    prompts = json.loads((train_dir / "prompts.json").read_text(encoding="utf-8"))
    train_latents = torch.load(train_dir / "train_latents.pt", map_location="cpu")["latents"]
    prompt_ids = torch.load(train_dir / "prompt_ids.pt", map_location="cpu")["prompt_ids"]

    embeds = torch.load(train_dir / "prompt_embeds.pt", map_location="cpu")
    cond_all = embeds["cond"]
    uncond_all = embeds["uncond"]

    student_unet_dir = Path(student_unet_dir)
    student.pipe.unet = UNet2DConditionModel.from_pretrained(
        student_unet_dir,
        torch_dtype=student.dtype,
    ).to(device)
    student.pipe.unet.eval()

    descriptor = build_descriptor(cfg.descriptor, device=device)

    train_imgs_cpu = []
    train_emb_chunks = []
    for i in tqdm(range(0, train_latents.shape[0], bs), desc="Embed trainset", leave=False):
        imgs = teacher.decode_latents_to_images_01(train_latents[i:i+bs].to(device=device, dtype=teacher.dtype))
        train_imgs_cpu.append(imgs.detach().cpu())
        train_emb_chunks.append(descriptor.encode(imgs).detach().cpu())
    train_imgs_cpu = torch.cat(train_imgs_cpu, dim=0)
    train_emb = torch.cat(train_emb_chunks, dim=0).to(device)

    N = len(prompts)
    idx = torch.randint(0, N, (cfg.eval_samples,), device=device)
    eval_prompts = [prompts[int(i)] for i in idx.detach().cpu().tolist()]
    eval_cond = cond_all[idx.cpu()].to(device=device, dtype=teacher.dtype)
    eval_uncond = uncond_all[idx.cpu()].to(device=device, dtype=teacher.dtype)
    eval_cond_ids = idx.to(device=device, dtype=torch.long)

    init_xt = torch.randn(
        (cfg.eval_samples, teacher.latent_channels, teacher.latent_size, teacher.latent_size),
        device=device,
        dtype=teacher.dtype,
    )
    init_xt = init_xt * float(getattr(teacher.eval_scheduler, "init_noise_sigma", 1.0))

    teacher_imgs_cpu = []
    student_imgs_cpu = []
    for i in tqdm(range(0, cfg.eval_samples, bs), desc="Sample teacher/student"):
        z0 = init_xt[i:i+bs]
        cond = eval_cond[i:i+bs]
        uncond = eval_uncond[i:i+bs]

        _, t_imgs = teacher.sample_latents_and_images(
            cond_uncond_embeds=(cond, uncond),
            init_latents=z0,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
        )
        _, s_imgs = student.sample_latents_and_images(
            cond_uncond_embeds=(cond, uncond),
            init_latents=z0,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
        )

        teacher_imgs_cpu.append(t_imgs.detach().cpu())
        student_imgs_cpu.append(s_imgs.detach().cpu())

    teacher_imgs_cpu = torch.cat(teacher_imgs_cpu, dim=0)
    student_imgs_cpu = torch.cat(student_imgs_cpu, dim=0)

    pfd_ts = estimate_pfd(
        teacher_images_01=teacher_imgs_cpu.to(device),
        student_images_01=student_imgs_cpu.to(device),
        descriptor=descriptor,
    )

    fid_st = compute_fid(
        real_images_01=teacher_imgs_cpu,
        fake_images_01=student_imgs_cpu,
        device=device,
        batch_size=cfg.fid_batch_size,
    )

    m_dist = m_distance_to_trainset(
        student_images_01=student_imgs_cpu.to(device),
        trainset_emb=train_emb,
        descriptor=descriptor,
        chunk=bs,
    )

    fid_train = compute_fid(
        real_images_01=train_imgs_cpu,
        fake_images_01=student_imgs_cpu,
        device=device,
        batch_size=cfg.fid_batch_size,
    )

    emp_state = prepare_empirical_state(
        train_latents=train_latents,
        cond_ids=prompt_ids,
        device=device,
    )

    ddim = teacher.eval_scheduler
    ddim.set_timesteps(cfg.num_inference_steps, device=device)
    timesteps = ddim.timesteps
    alphas_cumprod = ddim.alphas_cumprod.to(device=device, dtype=torch.float32)

    pfd_se_vals = []
    for i in tqdm(range(0, cfg.eval_samples, bs), desc="PFD student vs empirical", leave=False):
        z0 = init_xt[i:i+bs]
        cid = eval_cond_ids[i:i+bs]

        emp_x0 = sample_empirical_ddim_latents(
            init_xt=z0,
            cond_ids=cid,
            alphas_cumprod=alphas_cumprod,
            timesteps=timesteps,
            state=emp_state,
            chunk_train=cfg.empirical_chunk_train,
        ).to(dtype=teacher.dtype)

        emp_imgs = teacher.decode_latents_to_images_01(emp_x0).detach()

        s_imgs = student_imgs_cpu[i:i+bs].to(device)

        emb_s = descriptor.encode(s_imgs)
        emb_e = descriptor.encode(emp_imgs)
        d = torch.norm(emb_s - emb_e, dim=-1)
        pfd_se_vals.append(d.detach().cpu())

    pfd_se_all = torch.cat(pfd_se_vals, dim=0)
    pfd_student_emp_mean = float(pfd_se_all.mean().item())
    pfd_student_emp_std = float(pfd_se_all.std(unbiased=False).item())
    pfd_student_emp_n = int(pfd_se_all.numel())

    out = {
        "train_size": int(train_latents.shape[0]),
        "eval_samples": int(cfg.eval_samples),
        "num_inference_steps": int(cfg.num_inference_steps),
        "guidance_scale": float(cfg.guidance_scale),
        "descriptor": str(cfg.descriptor),

        "pfd_teacher_student_mean": pfd_ts.pfd_mean,
        "pfd_teacher_student_std": pfd_ts.pfd_std,
        "pfd_teacher_student_n": pfd_ts.n,
        "fid_student_teacher": fid_st.fid,

        "pfd_student_empirical_mean": pfd_student_emp_mean,
        "pfd_student_empirical_std": pfd_student_emp_std,
        "pfd_student_empirical_n": pfd_student_emp_n,
        "m_distance_student_to_train_mean": float(m_dist),
        "fid_student_train": fid_train.fid,
    }

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
