from dataclasses import dataclass
from typing import Dict, Tuple
import torch


@dataclass
class EmpiricalSamplerState:
    c_latents: Dict[int, torch.Tensor] # cond_id -> (N_c,C,H,W)
    c_flat: Dict[int, torch.Tensor] # cond_id -> (N_c,D)
    c_norm2: Dict[int, torch.Tensor] # cond_id -> (N_c,)
    cond_present: torch.Tensor # (K,)


def prepare_empirical_state(
    train_latents: torch.Tensor,
    cond_ids: torch.Tensor,
    device: torch.device,
) -> EmpiricalSamplerState:
    train_latents = train_latents.to(device=device, dtype=torch.float32)
    cond_ids = cond_ids.to(device=device, dtype=torch.long)

    c_latents: Dict[int, torch.Tensor] = {}
    c_flat: Dict[int, torch.Tensor] = {}
    c_norm2: Dict[int, torch.Tensor] = {}

    uniq = torch.unique(cond_ids).detach().cpu().tolist()
    for cid in uniq:
        idx = (cond_ids == cid).nonzero(as_tuple=False).squeeze(-1)
        yl = train_latents[idx]
        yf = yl.flatten(1)
        yn2 = (yf * yf).sum(dim=1)
        c_latents[int(cid)] = yl
        c_flat[int(cid)] = yf
        c_norm2[int(cid)] = yn2

    return EmpiricalSamplerState(
        c_latents=c_latents,
        c_flat=c_flat,
        c_norm2=c_norm2,
        cond_present=torch.tensor(uniq, device=device, dtype=torch.long),
    )


@torch.no_grad()
def sample_empirical_ddim_latents(
    *,
    init_xt: torch.Tensor, # (B,C,H,W)
    cond_ids: torch.Tensor, # (B,)
    alphas_cumprod: torch.Tensor, # (T,)
    timesteps: torch.Tensor, # (S,) decreasing
    state: EmpiricalSamplerState,
    chunk_train: int = 1024,
) -> torch.Tensor:
    """
    Deterministic DDIM-style sampling from the *empirical* distribution:
    x0 ~ uniform over training latents for that cond_id.

    Mixture posterior weights:
      w_i ∝ exp(-||x_t - a y_i||^2 / (2 sigma^2))
    and posterior mean mu = E[x0 | x_t] used as DDIM x0_pred.
    """
    device = init_xt.device
    x = init_xt.to(device=device, dtype=torch.float32)
    cond_ids = cond_ids.to(device=device, dtype=torch.long)

    B = x.shape[0]
    D = x[0].numel()

    for si, t in enumerate(timesteps):
        t_int = int(t.item())
        a_bar = alphas_cumprod[t_int].clamp(0.0, 1.0)
        a = torch.sqrt(a_bar)
        sigma2 = (1.0 - a_bar).clamp(min=1e-12)

        last = (si == timesteps.numel() - 1)
        if not last:
            t_prev = int(timesteps[si + 1].item())
            a_bar_prev = alphas_cumprod[t_prev].clamp(0.0, 1.0)
            a_prev = torch.sqrt(a_bar_prev)
            sigma_prev = torch.sqrt((1.0 - a_bar_prev).clamp(min=0.0))
        else:
            a_prev = None
            sigma_prev = None

        x_flat = x.flatten(1)
        x_norm2 = (x_flat * x_flat).sum(dim=1)
        mu = torch.zeros_like(x, dtype=torch.float32)

        for cid in torch.unique(cond_ids).detach().cpu().tolist():
            cid = int(cid)
            idx_b = (cond_ids == cid).nonzero(as_tuple=False).squeeze(-1)
            if idx_b.numel() == 0:
                continue

            if cid not in state.c_flat:
                cid = int(state.cond_present[0].item())

            y_f = state.c_flat[cid].to(device=device, dtype=torch.float32)
            y_n2 = state.c_norm2[cid].to(device=device, dtype=torch.float32)

            xb = x_flat[idx_b]
            xb_n2 = x_norm2[idx_b]
            bsz = xb.shape[0]

            max_logit = torch.full((bsz,), -1e30, device=device, dtype=torch.float32)
            for j in range(0, y_f.shape[0], chunk_train):
                yfj = y_f[j:j + chunk_train]
                y_n2j = y_n2[j:j + chunk_train]
                dot = xb @ yfj.T
                dist2 = xb_n2[:, None] + (a * a) * y_n2j[None, :] - 2.0 * a * dot
                logits = -dist2 / (2.0 * sigma2)
                max_logit = torch.maximum(max_logit, logits.max(dim=1).values)

            denom = torch.zeros((bsz,), device=device, dtype=torch.float32)
            mu_flat = torch.zeros((bsz, D), device=device, dtype=torch.float32)

            for j in range(0, y_f.shape[0], chunk_train):
                yfj = y_f[j:j + chunk_train]
                y_n2j = y_n2[j:j + chunk_train]
                dot = xb @ yfj.T
                dist2 = xb_n2[:, None] + (a * a) * y_n2j[None, :] - 2.0 * a * dot
                logits = -dist2 / (2.0 * sigma2)
                w = torch.exp(logits - max_logit[:, None])
                denom = denom + w.sum(dim=1)
                mu_flat = mu_flat + (w @ yfj)

            denom = denom.clamp(min=1e-12)
            mu_flat = mu_flat / denom[:, None]
            mu[idx_b] = mu_flat.view(bsz, *x.shape[1:])

        if last:
            x = mu
            break

        sigma = torch.sqrt(sigma2)
        eps_hat = (x - a * mu) / sigma
        x = a_prev * mu + sigma_prev * eps_hat

    return x
