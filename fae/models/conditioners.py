from typing import Sequence

import torch
import torch.nn as nn
from fae.generators.common import ConditioningBundle


class ClassConditioner(nn.Module):
    def __init__(self, num_classes: int, dim: int) :
        super().__init__()
        self.embedding = nn.Embedding(num_classes, dim)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embedding(labels)


class FrozenTextConditioner(nn.Module):
    def __init__(self, model_name: str = "google-t5/t5-base", out_dim: int = 1024, max_length: int = 64) :
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:
            raise ImportError("transformers is required for FrozenTextConditioner.") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        hidden = int(self.encoder.config.hidden_size)
        self.proj = nn.Linear(hidden, out_dim)
        self.max_length = max_length

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str], device: torch.device) -> torch.Tensor:
        tokens = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        outputs = self.encoder(**tokens)
        hidden = outputs.last_hidden_state
        pooled = hidden.mean(dim=1)
        return self.proj(pooled)

    def forward(self, texts: Sequence[str], device: torch.device) -> torch.Tensor:
        return self.encode_texts(texts, device=device)


def build_internal_conditioning(
    labels: list[int | None] | None,
    captions: list[str | None] | None,
    device: torch.device,
    class_conditioner: ClassConditioner | None = None,
    text_conditioner: FrozenTextConditioner | None = None,
) -> ConditioningBundle | None:
    bundle = ConditioningBundle()
    has_any = False
    if class_conditioner is not None and labels is not None and all(label is not None for label in labels):
        label_tensor = torch.tensor(labels, device=device, dtype=torch.long)
        bundle.class_labels = label_tensor
        bundle.vector = class_conditioner(label_tensor)
        has_any = True
    if text_conditioner is not None and captions is not None and all(caption is not None for caption in captions):
        text_vec = text_conditioner(captions, device=device)
        bundle.vector = text_vec if bundle.vector is None else bundle.vector + text_vec
        has_any = True
    return bundle if has_any else None
