from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from .data import save_latents
from .utils.seed import seed_everything
from .sampling import sample_dit_latents_and_images, _infer_num_cond_classes


@dataclass
class TeacherDatasetConfig:
    out_path: Path
    train_size: int
    num_inference_steps: int = 30
    guidance_scale: float = 4.0
    seed: int = 0
    batch_size: int = 8


@torch.no_grad()
def generate_teacher_latents(
    teacher_pipe,
    cfg: TeacherDatasetConfig,
    *,
    device: Optional[torch.device] = None,
) -> None:
    """
    Sample synthetic train latents x0 from teacher (teacher–student protocol).
    """
    device = device or teacher_pipe.device
    seed_everything(cfg.seed)

    n_cond = _infer_num_cond_classes(teacher_pipe)
    labels = torch.randint(low=0, high=n_cond, size=(cfg.train_size,), device=device, dtype=torch.long)

    out_latents = []
    out_labels = []

    bs = int(cfg.batch_size)

    for i in tqdm(range(0, cfg.train_size, bs), desc="Sampling teacher trainset"):
        l = labels[i : i + bs]

        out = sample_dit_latents_and_images(
            pipe=teacher_pipe,
            class_labels=l,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            latents=None,
            generator=None,
        )
        out_latents.append(out.latents.detach().float().cpu())
        out_labels.append(l.detach().cpu())

    lat = torch.cat(out_latents, dim=0)
    lab = torch.cat(out_labels, dim=0)

    save_latents(cfg.out_path, lat, lab)
