from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F

from fae.utils.image import infer_hw_from_tokens, infer_prefix_tokens


@dataclass
class BackboneFeatures:
    tokens: torch.Tensor
    spatial_shape: tuple[int, int]
    prefix_tokens: int
    raw_sequence: torch.Tensor


class FrozenVisionBackbone(nn.Module):
    model_name: str
    output_dim: int
    patch_size: int | None = None
    input_size: int | None = None

    def preprocess(self, images: list[Any]) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def build_reconstruction_targets(self, images: list[Any], output_size: int | tuple[int, int] | None = None) -> torch.Tensor:
        """Return RGB targets in [0, 1] for reconstruction loss"""
        raise NotImplementedError

    @property
    def latent_hw(self) -> tuple[int, int] | None:
        if self.input_size is None or self.patch_size is None:
            return None
        return self.input_size // self.patch_size, self.input_size // self.patch_size

    def _processor_kwargs(self) -> dict[str, Any]:
        if self.input_size is None:
            return {}
        size = {"height": int(self.input_size), "width": int(self.input_size)}
        return {"size": size, "crop_size": size}

    def _raw_to_reconstruction_targets(self, images: list[Any], output_size: int | tuple[int, int] | None = None) -> torch.Tensor:
        if output_size is None:
            output_size = self.input_size
        if output_size is None:
            raise ValueError("output_size must be provided when backbone.input_size is unknown.")
        if isinstance(output_size, int):
            size = (int(output_size), int(output_size))
        else:
            size = (int(output_size[0]), int(output_size[1]))

        processed: list[torch.Tensor] = []
        for image in images:
            if isinstance(image, Image.Image):
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image = image.resize((size[1], size[0]), resample=Image.BICUBIC)
                tensor = torch.from_numpy(__import__("numpy").array(image)).permute(2, 0, 1).to(dtype=torch.float32).div_(255.0)
            elif isinstance(image, torch.Tensor):
                tensor = image.detach().cpu().to(dtype=torch.float32)
                if tensor.ndim == 4 and tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)
                if tensor.ndim != 3:
                    raise ValueError(f"Expected image tensor with 3 dims, got shape {tuple(tensor.shape)}")
                if tensor.shape[0] not in {1, 3} and tensor.shape[-1] in {1, 3}:
                    tensor = tensor.permute(2, 0, 1).contiguous()
                if float(tensor.max()) > 1.5:
                    tensor = tensor / 255.0
                elif float(tensor.min()) < -0.05 or float(tensor.max()) > 1.05:
                    tensor = tensor.clamp(-1.0, 1.0).add(1.0).mul(0.5)
                else:
                    tensor = tensor.clamp(0.0, 1.0)
            else:
                raise TypeError(f"Unsupported image type for reconstruction targets: {type(image)!r}")

            if tuple(tensor.shape[-2:]) != size:
                tensor = F.interpolate(tensor.unsqueeze(0), size=size, mode="bicubic", align_corners=False, antialias=True).squeeze(0)
            processed.append(tensor.clamp(0.0, 1.0))

        return torch.stack(processed, dim=0)

    def _processor_to_reconstruction_targets(self, processor: Any, images: list[Any], output_size: int | tuple[int, int] | None = None) -> torch.Tensor:
        if output_size is not None:
            return self._raw_to_reconstruction_targets(images, output_size=output_size)
        batch = processor(images=images, return_tensors="pt", do_normalize=False, **self._processor_kwargs())
        pixel_values = batch.get("pixel_values")
        if pixel_values is None:
            raise KeyError("Backbone processor did not return 'pixel_values'.")
        pixel_values = pixel_values.to(dtype=torch.float32)
        if float(pixel_values.max()) > 1.5:
            pixel_values = pixel_values / 255.0
        return pixel_values.clamp(0.0, 1.0)

    @torch.no_grad()
    def forward_features(self, inputs: dict[str, torch.Tensor]) -> BackboneFeatures:
        raise NotImplementedError

    @staticmethod
    def _split_patch_tokens(sequence: torch.Tensor, prefix_tokens: int | None = None) -> BackboneFeatures:
        if prefix_tokens is None:
            prefix_tokens = infer_prefix_tokens(sequence.shape[1])
        patch_tokens = sequence[:, prefix_tokens:]
        spatial = infer_hw_from_tokens(patch_tokens.shape[1])
        return BackboneFeatures(
            tokens=patch_tokens,
            spatial_shape=spatial,
            prefix_tokens=prefix_tokens,
            raw_sequence=sequence,
        )
