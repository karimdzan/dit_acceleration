from types import SimpleNamespace

import fae.backbones.dinov2 as dinov2_mod


class _DummyProcessor:
    size = {"height": 224, "width": 224}

    def __call__(self, *args, **kwargs):
        return {"pixel_values": None}


class _DummyModel:
    def __init__(self, config):
        self.config = config

    @classmethod
    def from_pretrained(cls, model_name):
        return cls(SimpleNamespace(hidden_size=768, patch_size=14, num_register_tokens=4))

    def eval(self):
        return self

    def parameters(self):
        return []


def test_dinov2_with_registers_uses_all_prefix_tokens(monkeypatch):
    monkeypatch.setattr(dinov2_mod, "_TRANSFORMERS_IMPORT_ERROR", None)
    monkeypatch.setattr(dinov2_mod, "AutoImageProcessor", SimpleNamespace(from_pretrained=lambda *args, **kwargs: _DummyProcessor()))
    monkeypatch.setattr(dinov2_mod, "Dinov2Config", SimpleNamespace(from_pretrained=lambda *args, **kwargs: SimpleNamespace(num_register_tokens=4)))
    monkeypatch.setattr(dinov2_mod, "Dinov2WithRegistersModel", _DummyModel)
    monkeypatch.setattr(dinov2_mod, "Dinov2Model", _DummyModel)

    backbone = dinov2_mod.DINOv2Backbone(model_name="facebook/dinov2-with-registers-base", prefix_tokens=None, input_size=224)
    assert backbone.prefix_tokens == 5
