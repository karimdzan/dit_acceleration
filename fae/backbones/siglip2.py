from typing import Any

import torch
from transformers import AutoImageProcessor, AutoModel

from .base import BackboneFeatures, FrozenVisionBackbone


class SigLIP2Backbone(FrozenVisionBackbone):
    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
        processor_name: str | None = None,
        prefix_tokens: int | None = 0,
        local_files_only: bool = True,
        use_fast_processor: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        processor_name = processor_name or model_name

        self.model_name = model_name
        self.prefix_tokens = prefix_tokens

        self.processor = AutoImageProcessor.from_pretrained(
            processor_name,
            local_files_only=local_files_only,
            use_fast=use_fast_processor,
        )

        base = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )

        self.model = base.vision_model if hasattr(base, "vision_model") else base
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False

        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None and hasattr(base, "config") and hasattr(base.config, "vision_config"):
            hidden_size = getattr(base.config.vision_config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Could not infer hidden_size from SigLIP backbone config.")
        self.output_dim = int(hidden_size)

        self.patch_size = int(getattr(self.model.config, "patch_size", 16))

    def preprocess(self, images: list[Any]) -> dict[str, torch.Tensor]:
        batch = self.processor(images=images, return_tensors="pt")
        return {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}

    @torch.no_grad()
    def forward_features(self, inputs: dict[str, torch.Tensor]) -> BackboneFeatures:
        outputs = self.model(
            **inputs,
            output_hidden_states=False,
            return_dict=True,
        )

        if not hasattr(outputs, "last_hidden_state"):
            raise RuntimeError("SigLIP backbone output does not contain last_hidden_state.")

        sequence = outputs.last_hidden_state
        return self._split_patch_tokens(sequence, self.prefix_tokens)