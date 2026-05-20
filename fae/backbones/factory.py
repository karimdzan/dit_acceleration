def build_backbone(name: str, **kwargs):
    key = name.lower()
    if key == "dinov2":
        from .dinov2 import DINOv2Backbone
        return DINOv2Backbone(**kwargs)
    if key == "siglip2":
        from .siglip2 import SigLIP2Backbone
        return SigLIP2Backbone(**kwargs)
    if key in {"vit_mae", "mae"}:
        from .vit_mae import ViTMAEBackbone
        return ViTMAEBackbone(**kwargs)
    raise KeyError(f"Unknown backbone: {name}. Available: ['dinov2', 'siglip2', 'vit_mae', 'mae']")
