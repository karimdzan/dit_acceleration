from typing import Any

import torch
try:
    from transformers import AutoImageProcessor, AutoModel
except ModuleNotFoundError as exc:
    AutoImageProcessor = None
    AutoModel = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None

from fae.utils.hf_local import HFLocalResolutionError, resolve_hf_repo_root

from .base import BackboneFeatures, FrozenVisionBackbone


def _call_from_pretrained(factory, source, **kwargs):
    try:
        return factory.from_pretrained(source, **kwargs)
    except TypeError:
        return factory.from_pretrained(source)


class SigLIP2Backbone(FrozenVisionBackbone):
    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
        processor_name: str | None = None,
        prefix_tokens: int | None = 0,
        local_files_only: bool = True,
        use_fast_processor: bool = False,
        input_size: int | None = None,
        encoder_input_size: int | None = None,
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

        processor_name = processor_name or model_name
        self.model_name = model_name
        self.prefix_tokens = prefix_tokens
        repo_root = repo_root or local_dir
        model_source: str | Any = model_name
        processor_source: str | Any = processor_name
        if snapshot_dir or repo_root or cache_dir:
            try:
                model_resolved = resolve_hf_repo_root(
                    model_name,
                    snapshot_dir=snapshot_dir,
                    repo_root=repo_root,
                    revision=revision,
                    cache_dir=cache_dir,
                    require_local=local_files_only,
                )
                processor_resolved = resolve_hf_repo_root(
                    processor_name,
                    snapshot_dir=snapshot_dir if processor_name == model_name else None,
                    repo_root=repo_root if processor_name == model_name else None,
                    revision=revision,
                    cache_dir=cache_dir,
                    require_local=local_files_only,
                )
            except HFLocalResolutionError:
                if local_files_only:
                    raise
                model_resolved = None
                processor_resolved = None
            if model_resolved is not None:
                model_source = str(model_resolved)
            if processor_resolved is not None:
                processor_source = str(processor_resolved)
        self.processor = _call_from_pretrained(
            AutoImageProcessor,
            processor_source,
            local_files_only=local_files_only,
            use_fast=use_fast_processor,
        )
        base = _call_from_pretrained(
            AutoModel,
            model_source,
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
        self.patch_size = int(kwargs.get("patch_size", getattr(self.model.config, "patch_size", 16)))
        proc_size = getattr(self.processor, "size", {})
        default_size = proc_size.get("height", proc_size.get("shortest_edge", 256)) if isinstance(proc_size, dict) else 256
        self.input_size = int(input_size or encoder_input_size or kwargs.get("image_size") or default_size)

    def preprocess(self, images: list[Any]) -> dict[str, torch.Tensor]:
        batch = self.processor(images=images, return_tensors="pt", **self._processor_kwargs())
        return {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}

    def build_reconstruction_targets(self, images: list[Any], output_size: int | tuple[int, int] | None = None) -> torch.Tensor:
        return self._processor_to_reconstruction_targets(self.processor, images, output_size=output_size)

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
