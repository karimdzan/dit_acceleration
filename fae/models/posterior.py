from dataclasses import dataclass

import torch


@dataclass
class DiagonalGaussianPosterior:
    mean: torch.Tensor
    logvar: torch.Tensor

    def sample(self) -> torch.Tensor:
        std = torch.exp(0.5 * self.logvar)
        eps = torch.randn_like(std)
        return self.mean + std * eps

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self) -> torch.Tensor:
        kl = -0.5 * (1 + self.logvar - self.mean.pow(2) - self.logvar.exp())
        return kl.mean()
