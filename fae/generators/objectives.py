import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .common import ConditioningBundle, LossOutput


@dataclass
class DiffusionSchedule:
    alpha: torch.Tensor
    sigma: torch.Tensor


class SimpleCosineDiffusionObjective:
    def __init__(self, prediction_type: str = "v_prediction") :
        self.prediction_type = prediction_type

    def build_schedule(self, t: torch.Tensor, time_shift: float = 0.0) -> DiffusionSchedule:
        t = (t + time_shift).clamp(0.0, 1.0)
        angle = t * math.pi / 2.0
        alpha = torch.cos(angle)
        sigma = torch.sin(angle)
        return DiffusionSchedule(alpha=alpha, sigma=sigma)

    def training_loss(self, model, x0: torch.Tensor, conditioning: ConditioningBundle | None = None, time_shift: float = 0.0) -> LossOutput:
        b = x0.shape[0]
        t = torch.rand(b, device=x0.device)
        noise = torch.randn_like(x0)
        sched = self.build_schedule(t, time_shift=time_shift)
        alpha = sched.alpha[:, None, None, None].to(x0.dtype)
        sigma = sched.sigma[:, None, None, None].to(x0.dtype)
        x_t = alpha * x0 + sigma * noise
        pred = model(x_t, t, conditioning)

        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "sample":
            target = x0
        else:
            target = alpha * noise - sigma * x0
        if pred.shape[1] == target.shape[1] * 2:
            pred = pred[:, :target.shape[1]]
        loss = F.mse_loss(pred.float(), target.float())
        return LossOutput(loss=loss, logs={"loss": float(loss.detach().cpu())})

    @torch.no_grad()
    def sample(self, model, latent_shape: tuple[int, int, int, int], device: torch.device, conditioning: ConditioningBundle | None = None, num_steps: int = 30, time_shift: float = 0.0) -> torch.Tensor:
        z = torch.randn(latent_shape, device=device)
        ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        for i in range(num_steps):
            t = torch.full((latent_shape[0],), ts[i].item(), device=device)
            next_t = torch.full((latent_shape[0],), ts[i + 1].item(), device=device)
            sched = self.build_schedule(t, time_shift=time_shift)
            next_sched = self.build_schedule(next_t, time_shift=time_shift)
            pred = model(z, t, conditioning)
            if pred.shape[1] == z.shape[1] * 2:
                pred = pred[:, : z.shape[1]]
            alpha = sched.alpha[:, None, None, None].to(z.dtype)
            sigma = sched.sigma[:, None, None, None].to(z.dtype)
            if self.prediction_type == "epsilon":
                eps = pred
                x0 = (z - sigma * eps) / alpha.clamp_min(1e-6)
            elif self.prediction_type == "sample":
                x0 = pred
                eps = (z - alpha * x0) / sigma.clamp_min(1e-6)
            else:
                x0 = alpha * z - sigma * pred
                eps = sigma * z + alpha * pred
            z = next_sched.alpha[:, None, None, None].to(z.dtype) * x0 + next_sched.sigma[:, None, None, None].to(z.dtype) * eps
        return z


class FlowMatchingObjective:
    def training_loss(self, model, x0: torch.Tensor, conditioning: ConditioningBundle | None = None) -> LossOutput:
        b = x0.shape[0]
        t = torch.rand(b, device=x0.device)
        noise = torch.randn_like(x0)
        x_t = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * noise
        target = noise - x0
        pred = model(x_t, t, conditioning)
        loss = F.mse_loss(pred.float(), target.float())
        return LossOutput(loss=loss, logs={"loss": float(loss.detach().cpu())})

    @torch.no_grad()
    def sample(self, model, latent_shape: tuple[int, int, int, int], device: torch.device, conditioning: ConditioningBundle | None = None, num_steps: int = 30) -> torch.Tensor:
        x = torch.randn(latent_shape, device=device)
        ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        for i in range(num_steps):
            t = torch.full((latent_shape[0],), ts[i].item(), device=device)
            dt = ts[i + 1] - ts[i]
            v = model(x, t, conditioning)
            x = x + dt * v
        return x
