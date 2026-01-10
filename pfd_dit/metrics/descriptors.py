from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F

DescriptorName = Literal["identity", "clip"]

@dataclass
class Descriptor:
    name: str

    @torch.no_grad()
    def encode(self, images_01: torch.Tensor) -> torch.Tensor:
        '''
        images_01: float tensor in [0,1], shape (B,3,H,W)
        returns: (B,D) embeddings (L2-normalized)
        '''
        raise NotImplementedError

@dataclass
class IdentityDescriptor(Descriptor):
    '''Pixel-space descriptor: flatten normalized pixels.'''
    def __init__(self):
        super().__init__(name="identity")

    @torch.no_grad()
    def encode(self, images_01: torch.Tensor) -> torch.Tensor:
        x = images_01.float().view(images_01.shape[0], -1)
        return F.normalize(x, dim=-1)

@dataclass
class CLIPDescriptor(Descriptor):
    '''
    CLIP vision encoder descriptor. Uses `openai/clip-vit-base-patch32` by default.
    '''
    model_id: str = "openai/clip-vit-base-patch32"
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float16

    def __init__(self, device: str):
        super().__init__(name="clip")
        from transformers import CLIPVisionModel, CLIPImageProcessor

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPImageProcessor.from_pretrained(self.model_id)
        self.vision = CLIPVisionModel.from_pretrained(self.model_id).to(self.device)
        self.vision.eval()

        if self.device.type == "cuda":
            self.vision.to(self.dtype)

    @torch.no_grad()
    def encode(self, images_01: torch.Tensor) -> torch.Tensor:
        x = images_01.detach().clamp(0, 1)
        x_nhwc = x.permute(0, 2, 3, 1)  # B,H,W,C

        inputs = self.processor(images=[img.cpu().numpy() for img in x_nhwc], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        out = self.vision(**inputs)
        emb = out.pooler_output.float()
        return F.normalize(emb, dim=-1)

def build_descriptor(name: DescriptorName, *, device: Optional[torch.device] = None) -> Descriptor:
    if name == "identity":
        return IdentityDescriptor()
    if name == "clip":
        return CLIPDescriptor(device=device)
    raise ValueError(f"Unknown descriptor: {name}")
