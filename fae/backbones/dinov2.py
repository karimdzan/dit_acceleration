from typing import Any

import torch
try:
    from transformers import AutoImageProcessor, Dinov2Config, Dinov2Model
except ModuleNotFoundError as exc:
    AutoImageProcessor = None
    Dinov2Config = None
    Dinov2Model = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None


try:
    from transformers import Dinov2WithRegistersModel
except Exception:
    Dinov2WithRegistersModel = None

from fae.utils.hf_local import HFLocalResolutionError, resolve_hf_repo_root

from .base import BackboneFeatures, FrozenVisionBackbone


def _call_from_pretrained(factory, source, **kwargs):
    try:
        return factory.from_pretrained(source, **kwargs)
    except TypeError:
        return factory.from_pretrained(source)


class DINOv2Backbone(FrozenVisionBackbone):
    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        prefix_tokens: int | None = None,
        input_size: int | None = None,
        encoder_input_size: int | None = None,
        local_files_only: bool = True,
        revision: str | None = None,
        cache_dir: str | None = None,
        snapshot_dir: str | None = None,
        repo_root: str | None = None,
        local_dir: str | None = None,
        **kwargs,
    ) :
        super().__init__()
        if _TRANSFORMERS_IMPORT_ERROR is not None:
            raise ModuleNotFoundError("transformers is required for this backbone") from _TRANSFORMERS_IMPORT_ERROR
        self.model_name = model_name
        repo_root = repo_root or local_dir
        source: str | Any = model_name
        if snapshot_dir or repo_root or cache_dir:
            try:
                resolved = resolve_hf_repo_root(
                    model_name,
                    snapshot_dir=snapshot_dir,
                    repo_root=repo_root,
                    revision=revision,
                    cache_dir=cache_dir,
                    require_local=local_files_only,
                )
            except HFLocalResolutionError:
                if local_files_only:
                    raise
                resolved = None
            if resolved is not None:
                source = str(resolved)
        self.processor = _call_from_pretrained(AutoImageProcessor, source, local_files_only=local_files_only)
        config = _call_from_pretrained(Dinov2Config, source, local_files_only=local_files_only)
        num_register_tokens = int(getattr(config, "num_register_tokens", 0) or 0)
        use_register_model = num_register_tokens > 0 and Dinov2WithRegistersModel is not None
        model_cls = Dinov2WithRegistersModel if use_register_model else Dinov2Model
        self.model = _call_from_pretrained(model_cls, source, local_files_only=local_files_only)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.output_dim = int(self.model.config.hidden_size)
        default_prefix_tokens = 1 + int(getattr(self.model.config, "num_register_tokens", 0) or 0)
        self.prefix_tokens = int(prefix_tokens) if prefix_tokens is not None else default_prefix_tokens
        self.patch_size = int(kwargs.get("patch_size", getattr(self.model.config, "patch_size", 14)))
        self.input_size = int(input_size or encoder_input_size or kwargs.get("image_size") or getattr(self.processor, "size", {}).get("height", 224))

    def preprocess(self, images: list[Any]) -> dict[str, torch.Tensor]:
        batch = self.processor(images=images, return_tensors="pt", **self._processor_kwargs())
        return {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}

    def build_reconstruction_targets(self, images: list[Any], output_size: int | tuple[int, int] | None = None) -> torch.Tensor:
        return self._processor_to_reconstruction_targets(self.processor, images, output_size=output_size)

    @torch.no_grad()
    def forward_features(self, inputs: dict[str, torch.Tensor]) -> BackboneFeatures:
        outputs = self.model(**inputs)
        return self._split_patch_tokens(outputs.last_hidden_state, self.prefix_tokens)
