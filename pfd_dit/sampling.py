from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class SampleOutput:
    latents: torch.Tensor # final x0 latents (B,C,H,W)
    images_01: torch.Tensor # decoded images in [0,1] (B,3,H,W)


@torch.no_grad()
def decode_latents_to_images_01(pipe, latents: torch.Tensor) -> torch.Tensor:
    """
    Decode VAE latents -> images in [0,1].
    """
    imgs = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
    imgs = (imgs.clamp(-1, 1) + 1) / 2
    return imgs


def _infer_uncond_label(pipe) -> int:
    """
    For DiT CFG, diffusers uses a "null" class id appended after ImageNet classes.
    We infer it as (num_classes - 1). If num_classes==1000, guidance should be disabled.
    """
    n = getattr(pipe.transformer.config, "num_classes", None)
    if n is None:
        return 1000
    return int(n - 1)


def _infer_num_cond_classes(pipe) -> int:
    """
    Conditional labels are expected to be 0..(uncond_label-1).
    """
    uncond = _infer_uncond_label(pipe)
    return int(uncond)


@torch.no_grad()
def sample_dit_latents_and_images(
    *,
    pipe,
    class_labels: torch.Tensor, # (B,)
    num_inference_steps: int,
    guidance_scale: float = 4.0,
    latents: Optional[torch.Tensor] = None, # initial noise (B,C,H,W)
    generator: Optional[torch.Generator] = None,
) -> SampleOutput:
    """
    Manual DiT sampling so we can:
      - inject initial noise latents (for shared-noise PFD)
      - return final denoised latents (for building synthetic trainset)
    """
    device = pipe.device
    dtype = pipe.transformer.dtype

    class_labels = class_labels.to(device=device, dtype=torch.long)

    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    c = int(pipe.transformer.config.in_channels)
    h = int(pipe.transformer.config.sample_size)
    w = int(pipe.transformer.config.sample_size)

    bsz = int(class_labels.shape[0])

    if latents is None:
        latents = torch.randn((bsz, c, h, w), generator=generator, device=device, dtype=dtype)
        latents = latents * scheduler.init_noise_sigma
    else:
        latents = latents.to(device=device, dtype=dtype)

    do_cfg = guidance_scale is not None and float(guidance_scale) > 1.0
    if do_cfg:
        uncond_label = _infer_uncond_label(pipe)
        if getattr(pipe.transformer.config, "num_classes", 1001) <= uncond_label:
            do_cfg = False

    transformer = pipe.transformer
    transformer.eval()

    for t in timesteps:
        t_vec = torch.full((bsz,), int(t.item()), device=device, dtype=torch.long)

        if do_cfg:
            lat_in = torch.cat([latents, latents], dim=0)
            null = torch.full((bsz,), _infer_uncond_label(pipe), device=device, dtype=torch.long)
            cls_in = torch.cat([null, class_labels], dim=0)

            t_in = torch.cat([t_vec, t_vec], dim=0)

            model_out = transformer(lat_in, t_in, class_labels=cls_in).sample
            eps_uncond, eps_cond = model_out.chunk(2, dim=0)
            eps = eps_uncond + float(guidance_scale) * (eps_cond - eps_uncond)
        else:
            eps = transformer(latents, t_vec, class_labels=class_labels).sample

        c_lat = latents.shape[1]
        if eps.shape[1] != c_lat:
            eps = eps[:, :c_lat, :, :]

        step = scheduler.step(eps, t, latents)
        latents = step.prev_sample

    images_01 = decode_latents_to_images_01(pipe, latents)
    return SampleOutput(latents=latents, images_01=images_01)
