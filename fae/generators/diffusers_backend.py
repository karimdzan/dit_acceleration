import importlib
import inspect
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn

from .base import LatentGeneratorBackend
from .common import ConditioningBundle, LatentTensorSpec, LossOutput
from .objectives import FlowMatchingObjective, SimpleCosineDiffusionObjective


def _optional_import_diffusers():
    try:
        return importlib.import_module("diffusers")
    except Exception as exc:
        raise ImportError(
            "diffusers is required for external generator backends. Install with `pip install diffusers accelerate safetensors peft`."
        ) from exc


def _filter_supported_kwargs(fn, kwargs: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())

    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return {k: v for k, v in kwargs.items() if v is not None}

    accepted = {
        name
        for name, p in sig.parameters.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return {k: v for k, v in kwargs.items() if v is not None and k in accepted}

def _unwrap_sample(pred):
    if hasattr(pred, "sample"):
        return pred.sample
    if isinstance(pred, tuple):
        return pred[0]
    return pred


@dataclass
class DiffusersBackendConfig:
    kind: str
    transformer_name_or_path: str | None = None
    pipeline_name_or_path: str | None = None
    pipeline_class: str | None = None
    transformer_subfolder: str = "transformer"
    scheduler_subfolder: str = "scheduler"
    torch_dtype: str = "bfloat16"
    objective: str = "diffusion"
    prediction_type: str = "epsilon"
    train_mode: str = "frozen"
    guidance_dropout: float = 0.0
    sample_size: int | None = None
    in_channels: int | None = None


class DiffusersTransformerBackend(LatentGeneratorBackend):
    uses_native_prompt_encoder = True

    def __init__(self, cfg: DiffusersBackendConfig) :
        super().__init__()
        self.cfg = cfg
        self.diffusers = _optional_import_diffusers()
        self.pipeline = None
        self.model = None
        self.scheduler = None
        self._load_assets()
        self._configure_train_mode()
        objective = cfg.objective.lower()
        if cfg.kind == "diffusers_dit":
            self.objective = None
        else:
            self.objective = FlowMatchingObjective() if objective == "flow_matching" else SimpleCosineDiffusionObjective(prediction_type=cfg.prediction_type)

    def _dtype(self):
        return getattr(torch, self.cfg.torch_dtype)

    def _load_assets(self) :
        dtype = self._dtype()
        if self.cfg.pipeline_name_or_path:
            if not self.cfg.pipeline_class:
                raise ValueError("pipeline_class is required when pipeline_name_or_path is set.")
            pipeline_cls = getattr(self.diffusers, self.cfg.pipeline_class)
            self.pipeline = pipeline_cls.from_pretrained(self.cfg.pipeline_name_or_path, torch_dtype=dtype)
            self.model = getattr(self.pipeline, "transformer", None)
            self.scheduler = getattr(self.pipeline, "scheduler", None)
            if self.model is None:
                raise RuntimeError(f"Pipeline {self.cfg.pipeline_class} did not expose `.transformer`.")
        else:
            if not self.cfg.transformer_name_or_path:
                raise ValueError("Either transformer_name_or_path or pipeline_name_or_path must be set.")
            model_class_name = {
                "diffusers_dit": "DiTTransformer2DModel",
                "diffusers_sd3": "SD3Transformer2DModel",
                "diffusers_sana": "SanaTransformer2DModel",
            }[self.cfg.kind]
            model_cls = getattr(self.diffusers, model_class_name)
            self.model = model_cls.from_pretrained(
                self.cfg.transformer_name_or_path,
                subfolder=self.cfg.transformer_subfolder if self.cfg.transformer_subfolder else None,
                torch_dtype=dtype,
            )
            if self.cfg.kind == "diffusers_dit":
                scheduler_cls = getattr(self.diffusers, "DDIMScheduler")
                self.scheduler = scheduler_cls.from_pretrained(
                    self.cfg.transformer_name_or_path,
                    subfolder=self.cfg.scheduler_subfolder,
                )

    def _configure_train_mode(self) :
        mode = self.cfg.train_mode
        for p in self.model.parameters():
            p.requires_grad = mode != "frozen"
        if mode == "lora":
            try:
                from peft import LoraConfig, get_peft_model
                target_modules = ["to_q", "to_k", "to_v", "to_out.0", "q_proj", "k_proj", "v_proj", "o_proj"]
                lora_cfg = LoraConfig(r=16, lora_alpha=16, target_modules=target_modules, lora_dropout=0.0, bias="none")
                self.model = get_peft_model(self.model, lora_cfg)
            except Exception as exc:
                raise RuntimeError("Failed to enable LoRA. Install peft and ensure the backend modules expose supported attention projections.") from exc
        if hasattr(self.model, "enable_gradient_checkpointing"):
            self.model.enable_gradient_checkpointing()

    def latent_spec(self) -> LatentTensorSpec:
        cfg = self.model.config
        channels = int(self.cfg.in_channels or getattr(cfg, "in_channels", 4))
        size = int(self.cfg.sample_size or getattr(cfg, "sample_size", 32))
        return LatentTensorSpec(channels=channels, height=size, width=size)

    def _prepare_conditioning(self, conditioning: ConditioningBundle | None, device: torch.device) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if conditioning is not None:
            kwargs.update(
                class_labels=conditioning.class_labels,
                encoder_hidden_states=conditioning.encoder_hidden_states,
                pooled_projections=conditioning.pooled_projections,
                attention_mask=conditioning.attention_mask,
            )
            kwargs.update(conditioning.extra_kwargs)
        if self.cfg.kind == "diffusers_dit":
            num_classes = getattr(self.model.config, "num_embeds_ada_norm", None)
            if num_classes is not None:
                class_labels = kwargs.get("class_labels")
                if class_labels is None:
                    raise ValueError("Class-conditioned DiT backend requires ConditioningBundle.class_labels during training and sampling.")
                kwargs["class_labels"] = class_labels.to(device=device, dtype=torch.long)
        supported = _filter_supported_kwargs(self.model.forward, kwargs)
        return supported

    def _forward_model(self, x: torch.Tensor, t: torch.Tensor, conditioning: ConditioningBundle | None = None) -> torch.Tensor:
        kwargs: dict[str, Any] = {}

        if self.cfg.kind == "diffusers_dit":
            if conditioning is None or conditioning.class_labels is None:
                raise ValueError(
                    "diffusers_dit requires conditioning.class_labels, but None was provided. "
                    "Check dataset labels and PEFT kwargs filtering."
                )
            class_labels = conditioning.class_labels.to(device=x.device, dtype=torch.long)
            kwargs["class_labels"] = class_labels

        if conditioning is not None:
            if conditioning.encoder_hidden_states is not None:
                kwargs["encoder_hidden_states"] = conditioning.encoder_hidden_states
            if conditioning.pooled_projections is not None:
                kwargs["pooled_projections"] = conditioning.pooled_projections
            if conditioning.attention_mask is not None:
                kwargs["attention_mask"] = conditioning.attention_mask
            kwargs.update(conditioning.extra_kwargs)

        supported = _filter_supported_kwargs(self.model.forward, kwargs)
        pred = self.model(x, timestep=t, **supported)
        pred = _unwrap_sample(pred)

        if pred.shape[1] == x.shape[1] * 2:
            pred = pred[:, : x.shape[1]]
        
        return pred
        

    @torch.no_grad()
    def encode_prompts(self, prompts: Sequence[str], device: torch.device) -> ConditioningBundle:
        if self.pipeline is None:
            raise RuntimeError("Native prompt encoding requires loading a diffusers pipeline.")
        pipe = self.pipeline.to(device)
        call = pipe.encode_prompt
        kwargs = _filter_supported_kwargs(
            call,
            {
                "prompt": list(prompts),
                "device": device,
                "do_classifier_free_guidance": False,
                "num_images_per_prompt": 1,
            },
        )
        outputs = call(**kwargs)
        bundle = ConditioningBundle()
        if self.cfg.kind == "diffusers_sd3":
            if isinstance(outputs, tuple):
                bundle.encoder_hidden_states = outputs[0]
                if len(outputs) >= 3:
                    bundle.pooled_projections = outputs[2]
            else:
                bundle.encoder_hidden_states = outputs
        elif self.cfg.kind == "diffusers_sana":
            if isinstance(outputs, tuple):
                bundle.encoder_hidden_states = outputs[0]
                if len(outputs) >= 2 and isinstance(outputs[1], torch.Tensor):
                    bundle.attention_mask = outputs[1]
            else:
                bundle.encoder_hidden_states = outputs
        else:
            if isinstance(outputs, tuple):
                bundle.encoder_hidden_states = outputs[0] if isinstance(outputs[0], torch.Tensor) and outputs[0].ndim == 3 else None
                if bundle.encoder_hidden_states is None and len(outputs) > 1 and isinstance(outputs[1], torch.Tensor):
                    bundle.encoder_hidden_states = outputs[1]
            else:
                bundle.encoder_hidden_states = outputs
        return bundle.to(device)

    def training_loss(self, latents: torch.Tensor, conditioning: ConditioningBundle | None = None) -> LossOutput:
        if self.cfg.kind == "diffusers_dit":
            device = latents.device
            bsz = latents.shape[0]

            num_train_steps = int(getattr(self.scheduler.config, "num_train_timesteps", 1000)) if self.scheduler is not None else 1000
            timesteps = torch.randint(0, num_train_steps, (bsz,), device=device, dtype=torch.long)

            noise = torch.randn_like(latents)

            if self.scheduler is not None and hasattr(self.scheduler, "add_noise"):
                noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
                prediction_type = getattr(self.scheduler.config, "prediction_type", self.cfg.prediction_type)
            else:
                noisy_latents = latents + noise
                prediction_type = self.cfg.prediction_type

            pred = self._forward_model(noisy_latents, timesteps, conditioning)

            if prediction_type == "epsilon":
                target = noise
            elif prediction_type == "sample":
                target = latents
            elif prediction_type == "v_prediction":
                if self.scheduler is None or not hasattr(self.scheduler, "get_velocity"):
                    raise ValueError("v_prediction requires a scheduler with get_velocity().")
                target = self.scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unsupported prediction_type={prediction_type}")

            if pred.shape[1] == target.shape[1] * 2:
                pred = pred[:, : target.shape[1]]

            loss = nn.functional.mse_loss(pred.float(), target.float())
            return LossOutput(loss=loss, logs={"loss": float(loss.detach().cpu())})

        return self.objective.training_loss(self._forward_model, latents, conditioning)

    @torch.no_grad()
    def sample_latents(
        self,
        batch_size: int,
        device: torch.device,
        conditioning: ConditioningBundle | None = None,
        num_steps: int = 30,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        spec = self.latent_spec()
        shape = (batch_size, spec.channels, spec.height, spec.width)

        if self.cfg.kind == "diffusers_dit" and self.scheduler is not None:
            dtype = next(self.model.parameters()).dtype
            latents = torch.randn(shape, device=device, dtype=dtype)

            self.scheduler.set_timesteps(num_steps, device=device)
            for t in self.scheduler.timesteps:
                t_batch = torch.full((batch_size,), int(t), device=device, dtype=torch.long)
                pred = self._forward_model(latents, t_batch, conditioning)

                if pred.shape[1] == latents.shape[1] * 2:
                    pred = pred[:, : latents.shape[1]]

                latents = self.scheduler.step(pred, t, latents).prev_sample
            return latents

        return self.objective.sample(self._forward_model, shape, device, conditioning=conditioning, num_steps=num_steps)