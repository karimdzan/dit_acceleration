from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from pfd_dit.sampling import sample_dit_latents_and_images


@dataclass
class EmpiricalSamplerState:
    y_latents: Dict[int, torch.Tensor] # label -> (N_l, C, H, W)
    y_flat: Dict[int, torch.Tensor] # label -> (N_l, D)
    y_norm2: Dict[int, torch.Tensor] # label -> (N_l,)
    labels_present: torch.Tensor # (K,) unique labels present


def _prepare_empirical_state(
    train_latents: torch.Tensor,
    train_labels: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> EmpiricalSamplerState:
    """
    Prepare per-class caches for class-conditional empirical distribution p_emp(x | y).
    This matters because DiT is class-conditional; p_emp should match the label used
    during sampling.
    """
    train_latents = train_latents.to(device=device, dtype=torch.float32)
    train_labels = train_labels.to(device=device, dtype=torch.long)

    y_latents: Dict[int, torch.Tensor] = {}
    y_flat: Dict[int, torch.Tensor] = {}
    y_norm2: Dict[int, torch.Tensor] = {}

    uniq = torch.unique(train_labels).detach().cpu().tolist()
    for lab in uniq:
        idx = (train_labels == lab).nonzero(as_tuple=False).squeeze(-1)
        yl = train_latents[idx] # (N_l,C,H,W) float32
        yf = yl.flatten(1) # (N_l,D)
        yn2 = (yf * yf).sum(dim=1) # (N_l,)
        y_latents[int(lab)] = yl
        y_flat[int(lab)] = yf
        y_norm2[int(lab)] = yn2

    labels_present = torch.tensor(uniq, device=device, dtype=torch.long)
    return EmpiricalSamplerState(
        y_latents=y_latents,
        y_flat=y_flat,
        y_norm2=y_norm2,
        labels_present=labels_present,
    )


@torch.no_grad()
def sample_empirical_ddim_latents(
    *,
    init_xt: torch.Tensor, # (B,C,H,W) initial noise at t=T
    class_labels: torch.Tensor, # (B,) long
    alphas_cumprod: torch.Tensor, # (num_train_timesteps,) on device
    timesteps: torch.Tensor, # (num_inference_steps,) decreasing ints on device
    state: EmpiricalSamplerState,
    chunk_train: int = 1024,
) -> torch.Tensor:
    """
    Deterministic DDIM-like sampling from the empirical distribution p_emp(x | label)
    using closed-form posterior weights under the forward noising model.

    Forward model (DDPM-style):
      x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps

    If x0 is drawn from p_emp = uniform over {y_i}, then:
      p(x_t) is a Gaussian mixture with means sqrt(alpha_bar_t) y_i, variance (1-alpha_bar_t) I.
    We compute posterior weights over i and posterior mean E[x0 | x_t] exactly (up to softmax).
    """

    device = init_xt.device
    x = init_xt.to(device=device, dtype=torch.float32)
    labels = class_labels.to(device=device, dtype=torch.long)

    B = x.shape[0]
    D = x[0].numel()

    for si, t in enumerate(timesteps):
        t_int = int(t.item())
        a_bar = alphas_cumprod[t_int].clamp(0.0, 1.0)
        a = torch.sqrt(a_bar)
        sigma2 = (1.0 - a_bar).clamp(min=1e-12)

        last = (si == (timesteps.numel() - 1))
        if not last:
            t_prev = int(timesteps[si + 1].item())
            a_bar_prev = alphas_cumprod[t_prev].clamp(0.0, 1.0)
            a_prev = torch.sqrt(a_bar_prev)
            sigma2_prev = (1.0 - a_bar_prev).clamp(min=0.0)
            sigma_prev = torch.sqrt(sigma2_prev)
        else:
            a_prev = None
            sigma_prev = None

        x_flat = x.flatten(1) # (B,D)
        x_norm2 = (x_flat * x_flat).sum(dim=1) # (B,)

        mu = torch.zeros_like(x, dtype=torch.float32)

        for lab in torch.unique(labels).detach().cpu().tolist():
            lab = int(lab)
            mask = (labels == lab)
            idx_b = mask.nonzero(as_tuple=False).squeeze(-1)
            if idx_b.numel() == 0:
                continue

            if lab not in state.y_flat:
                lab_fallback = int(state.labels_present[0].item())
                y_f = state.y_flat[lab_fallback]
                y_n2 = state.y_norm2[lab_fallback]
                y_lat = state.y_latents[lab_fallback]
            else:
                y_f = state.y_flat[lab]
                y_n2 = state.y_norm2[lab]
                y_lat = state.y_latents[lab]

            xb = x_flat[idx_b] # (b,D)
            xb_n2 = x_norm2[idx_b] # (b,)

            # logits for mixture responsibilities:
            # log w_i ∝ -||x - a y_i||^2 / (2 sigma2)
            # expand: ||x||^2 + a^2||y||^2 - 2a<x,y>
            # compute dot in chunks over train items to reduce memory
            bsz = xb.shape[0]

            max_logit = torch.full((bsz,), -1e30, device=device, dtype=torch.float32)
            for j in range(0, y_f.shape[0], chunk_train):
                yfj = y_f[j:j + chunk_train].to(device=device, dtype=torch.float32) # (m,D)
                y_n2j = y_n2[j:j + chunk_train].to(device=device, dtype=torch.float32) # (m,)
                dot = xb @ yfj.T # (b,m)
                dist2 = xb_n2[:, None] + (a * a) * y_n2j[None, :] - 2.0 * a * dot
                logits = -dist2 / (2.0 * sigma2)
                max_logit = torch.maximum(max_logit, logits.max(dim=1).values)

            denom = torch.zeros((bsz,), device=device, dtype=torch.float32)
            mu_flat = torch.zeros((bsz, D), device=device, dtype=torch.float32)

            for j in range(0, y_f.shape[0], chunk_train):
                yfj = y_f[j:j + chunk_train].to(device=device, dtype=torch.float32) # (m,D)
                y_n2j = y_n2[j:j + chunk_train].to(device=device, dtype=torch.float32) # (m,)
                dot = xb @ yfj.T # (b,m)
                dist2 = xb_n2[:, None] + (a * a) * y_n2j[None, :] - 2.0 * a * dot
                logits = -dist2 / (2.0 * sigma2)
                w_unnorm = torch.exp(logits - max_logit[:, None]) # (b,m)

                denom = denom + w_unnorm.sum(dim=1)
                mu_flat = mu_flat + (w_unnorm @ yfj) # (b,D)

            denom = denom.clamp(min=1e-12)
            mu_flat = mu_flat / denom[:, None]
            mu_b = mu_flat.view(bsz, *x.shape[1:]) # (b,C,H,W)

            mu[idx_b] = mu_b

        if last:
            # DDIM end: output x0_pred = mu
            x = mu
            break

        # eps_hat = (x_t - a * mu) / sqrt(1-a_bar)
        sigma = torch.sqrt(sigma2)
        eps_hat = (x - a * mu) / sigma

        # deterministic update
        x = a_prev * mu + sigma_prev * eps_hat

    return x


@torch.no_grad()
def estimate_pfd_student_empirical(
    *,
    student_pipe,
    train_latents: torch.Tensor,
    train_labels: torch.Tensor,
    init_xt: torch.Tensor, # (B,C,H,W) shared initial noise at t=T
    class_labels: torch.Tensor, # (B,) shared labels
    ddim_scheduler,
    descriptor,
    num_inference_steps: int,
    guidance_scale: float,
    chunk_train: int = 1024,
) -> Tuple[float, float, int]:
    """
    Estimate PFD(p_theta, p_emp) via shared-noise deterministic mappings:
      - student mapping: standard DiT sampling (DDIM, eta=0) from init_xt
      - empirical mapping: deterministic empirical DDIM sampler above
    Then compute descriptor distance between resulting images.
    """

    device = init_xt.device
    dtype = student_pipe.transformer.dtype

    state = _prepare_empirical_state(train_latents, train_labels, device=device, dtype=dtype)

    ddim_scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = ddim_scheduler.timesteps
    alphas_cumprod = ddim_scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)

    old_sched = student_pipe.scheduler
    student_pipe.scheduler = ddim_scheduler
    try:
        student_out = sample_dit_latents_and_images(
            pipe=student_pipe,
            class_labels=class_labels,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            latents=init_xt.to(dtype=dtype),
        )
        student_imgs = student_out.images_01  # (B,3,H,W) in [0,1]
    finally:
        student_pipe.scheduler = old_sched

    emp_x0 = sample_empirical_ddim_latents(
        init_xt=init_xt,
        class_labels=class_labels,
        alphas_cumprod=alphas_cumprod,
        timesteps=timesteps,
        state=state,
        chunk_train=chunk_train,
    ).to(dtype=dtype)

    from pfd_dit.sampling import decode_latents_to_images_01
    emp_imgs = decode_latents_to_images_01(student_pipe, emp_x0)

    emb_s = descriptor.encode(student_imgs)
    emb_e = descriptor.encode(emp_imgs)
    d = torch.norm(emb_s - emb_e, dim=-1)

    return float(d.mean().item()), float(d.std(unbiased=False).item()), int(d.numel())
