from .dinov2 import DINOv2Backbone
from .siglip2 import SigLIP2Backbone
from .vit_mae import ViTMAEBackbone


_BACKBONES = {
    "dinov2": DINOv2Backbone,
    "siglip2": SigLIP2Backbone,
    "vit_mae": ViTMAEBackbone,
    "mae": ViTMAEBackbone,
}


def build_backbone(name: str, **kwargs):
    key = name.lower()
    if key not in _BACKBONES:
        raise KeyError(f"Unknown backbone: {name}. Available: {sorted(_BACKBONES)}")
    return _BACKBONES[key](**kwargs)
