from dataclasses import dataclass
from pathlib import Path
import json

import torch
from tqdm import tqdm
from diffusers import DDIMScheduler

from .utils.seed import seed_everything
from .metrics.descriptors import build_descriptor, DescriptorName
from .metrics.pfd import estimate_pfd, m_distance_to_trainset
from .metrics.fid import compute_fid
from .metrics.empirical_pfd import estimate_pfd_student_empirical
from .sampling import sample_dit_latents_and_images, decode_latents_to_images_01


@dataclass
class EvalConfig:
    out_path: Path
    eval_samples: int = 256
    num_inference_steps: int = 30
    guidance_scale: float = 4.0
    seed: int = 123
    descriptor: DescriptorName = "clip"
    batch_size: int = 8
    empirical_chunk_train: int = 1024


@torch.no_grad()
def evaluate_teacher_student(
    *,
    teacher_pipe,
    student_pipe,
    train_latents: torch.Tensor,
    train_labels: torch.Tensor,
    cfg: EvalConfig,
) -> dict:
    """
    Computes a paper-style suite of metrics:

    Generalization proxies (teacher ≈ p_data in teacher-student protocol):
      - PFD(student, teacher) under shared initial noise
      - FID(student, teacher)

    Memorization proxies (empirical distribution over the finite train set):
      - PFD(student, empirical) using an empirical DDIM sampler (approx Eq. 9)
      - M-distance(student samples -> train set) in descriptor space (baseline)
      - FID(student, train) (auxiliary / sanity)

    Notes:
      - We do not load real datasets here; teacher distribution is the p_data proxy.
      - We use a deterministic DDIM scheduler (eta=0 style) for shared-noise mappings.
    """
    seed_everything(cfg.seed)

    device = teacher_pipe.device
    bs = int(cfg.batch_size)

    descriptor = build_descriptor(cfg.descriptor, device=device)

    train_lat = train_latents.to(device)
    train_imgs_chunks = []
    train_emb_chunks = []
    for i in tqdm(range(0, train_lat.shape[0], bs), desc="Embedding trainset", leave=False):
        imgs = decode_latents_to_images_01(teacher_pipe, train_lat[i : i + bs])
        train_imgs_chunks.append(imgs.detach().cpu())
        train_emb_chunks.append(descriptor.encode(imgs).detach().cpu())
    train_imgs = torch.cat(train_imgs_chunks, dim=0).to(device)
    train_emb = torch.cat(train_emb_chunks, dim=0).to(device)

    base_cfg = getattr(getattr(teacher_pipe, "_train_scheduler", None), "config", None)
    if base_cfg is None:
        base_cfg = teacher_pipe.scheduler.config
    ddim = DDIMScheduler.from_config(base_cfg)

    uniq_train = torch.unique(train_labels.to(device)).to(device)
    if uniq_train.numel() == 0:
        raise RuntimeError("train_labels is empty; cannot evaluate class-conditional metrics.")
    labels = uniq_train[torch.randint(0, uniq_train.numel(), (cfg.eval_samples,), device=device)]

    c = int(teacher_pipe.transformer.config.in_channels)
    h = int(teacher_pipe.transformer.config.sample_size)
    w = int(teacher_pipe.transformer.config.sample_size)
    init_xt = torch.randn((cfg.eval_samples, c, h, w), device=device, dtype=teacher_pipe.transformer.dtype)
    init_xt = init_xt * float(getattr(ddim, "init_noise_sigma", 1.0))

    old_t_sched = teacher_pipe.scheduler
    old_s_sched = student_pipe.scheduler
    teacher_pipe.scheduler = ddim
    student_pipe.scheduler = ddim

    try:
        t_imgs_all, s_imgs_all = [], []
        for i in tqdm(range(0, cfg.eval_samples, bs), desc="Sampling teacher/student"):
            l = labels[i : i + bs]
            z0 = init_xt[i : i + bs]

            t_out = sample_dit_latents_and_images(
                pipe=teacher_pipe,
                class_labels=l,
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                latents=z0,
            )
            s_out = sample_dit_latents_and_images(
                pipe=student_pipe,
                class_labels=l,
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                latents=z0,
            )

            t_imgs_all.append(t_out.images_01.detach().cpu())
            s_imgs_all.append(s_out.images_01.detach().cpu())

        teacher_imgs = torch.cat(t_imgs_all, dim=0).to(device)
        student_imgs = torch.cat(s_imgs_all, dim=0).to(device)
    finally:
        teacher_pipe.scheduler = old_t_sched
        student_pipe.scheduler = old_s_sched

    pfd_ts = estimate_pfd(
        teacher_images_01=teacher_imgs,
        student_images_01=student_imgs,
        descriptor=descriptor,
    )

    torch.cuda.empty_cache()

    fid_bs = max(4, min(16, cfg.batch_size))
    fid_ts = compute_fid(
        real_images_01=teacher_imgs, 
        fake_images_01=student_imgs, 
        device=device,
        batch_size=fid_bs
    )

    m_dist = m_distance_to_trainset(
        student_images_01=student_imgs,
        trainset_emb=train_emb,
        descriptor=descriptor,
        chunk=bs,
    )
    
    torch.cuda.empty_cache()

    fid_train = compute_fid(
        real_images_01=train_imgs, 
        fake_images_01=student_imgs, 
        device=device,
        batch_size=fid_bs
    )

    pfd_se_means, pfd_se_stds = [], []
    total_n = 0
    for i in tqdm(range(0, cfg.eval_samples, bs), desc="PFD student vs empirical", leave=False):
        z0 = init_xt[i : i + bs]
        lab = labels[i : i + bs]

        mean_i, std_i, n_i = estimate_pfd_student_empirical(
            student_pipe=student_pipe,
            train_latents=train_latents,
            train_labels=train_labels,
            init_xt=z0,
            class_labels=lab,
            ddim_scheduler=ddim,
            descriptor=descriptor,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            chunk_train=int(cfg.empirical_chunk_train),
        )
        pfd_se_means.append(mean_i)
        pfd_se_stds.append(std_i)
        total_n += n_i

    pfd_student_emp_mean = float(torch.tensor(pfd_se_means).mean().item()) if pfd_se_means else float("nan")
    pfd_student_emp_std = float(torch.tensor(pfd_se_stds).mean().item()) if pfd_se_stds else float("nan")

    out = {
        "pfd_teacher_student_mean": pfd_ts.pfd_mean,
        "pfd_teacher_student_std": pfd_ts.pfd_std,
        "pfd_teacher_student_n": pfd_ts.n,
        "fid_student_teacher": fid_ts.fid,

        "pfd_student_empirical_mean": pfd_student_emp_mean,
        "pfd_student_empirical_std": pfd_student_emp_std,
        "pfd_student_empirical_n": int(total_n),
        "m_distance_student_to_train_mean": float(m_dist),
        "fid_student_train": fid_train.fid,

        "eval_samples": int(cfg.eval_samples),
        "num_inference_steps": int(cfg.num_inference_steps),
        "guidance_scale": float(cfg.guidance_scale),
        "descriptor": str(cfg.descriptor),
        "seed": int(cfg.seed),
        "batch_size": int(cfg.batch_size),
        "empirical_chunk_train": int(cfg.empirical_chunk_train),
        "train_size": int(train_latents.shape[0]),
    }

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return out
