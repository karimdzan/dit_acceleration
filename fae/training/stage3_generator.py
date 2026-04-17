import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fae.backbones.base import FrozenVisionBackbone
from fae.generators.base import LatentGeneratorBackend
from fae.models.conditioners import ClassConditioner, FrozenTextConditioner, build_internal_conditioning
from fae.models.fae import FeatureAutoEncoder
from fae.models.latent_bridge import BaseLatentAdapter
from fae.training.common import prepare_backbone_inputs
from fae.utils.distributed import is_main_process, reduce_mean_dict, unwrap_model


def _build_conditioning(
    backend: LatentGeneratorBackend,
    batch: dict,
    device: torch.device,
    class_conditioner: ClassConditioner | None = None,
    text_conditioner: FrozenTextConditioner | None = None,
):
    backend = unwrap_model(backend)
    if getattr(backend, "uses_native_prompt_encoder", False):
        labels = batch.get("labels")
        captions = batch.get("captions")
        cond = None
        if captions is not None and all(caption is not None for caption in captions):
            cond = backend.encode_prompts([str(c) for c in captions], device=device)
        if labels is not None and all(label is not None for label in labels):
            label_tensor = torch.tensor(labels, device=device, dtype=torch.long)
            if cond is None:
                from fae.generators.common import ConditioningBundle
                cond = ConditioningBundle(class_labels=label_tensor)
            else:
                cond.class_labels = label_tensor
        return cond
    return build_internal_conditioning(
        labels=batch.get("labels"),
        captions=batch.get("captions"),
        device=device,
        class_conditioner=class_conditioner,
        text_conditioner=text_conditioner,
    )


def train_stage3_epoch(
    backend: LatentGeneratorBackend,
    fae: FeatureAutoEncoder,
    bridge: BaseLatentAdapter,
    backbone: FrozenVisionBackbone,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    context,
    device: torch.device,
    class_conditioner: ClassConditioner | None = None,
    text_conditioner: FrozenTextConditioner | None = None,
    conditioner_optimizer: torch.optim.Optimizer | None = None,
    bridge_cycle_weight: float = 0.0,
    skip_oom_batches: bool = True,
) -> dict[str, float]:
    backend.train()
    bridge.train()
    fae.eval()
    backbone.eval()
    if class_conditioner is not None:
        class_conditioner.train()
    running = {"loss": 0.0, "backend_loss": 0.0, "bridge_cycle": 0.0}
    count = 0
    fae_param = next(fae.parameters(), None)
    fae_dtype = fae_param.dtype if fae_param is not None else torch.float32
    for batch in tqdm(dataloader, desc="stage3", leave=False, disable=not is_main_process()):
        if batch is None:
            continue
        images = batch["images"]
        try:
            backbone_inputs = prepare_backbone_inputs(backbone, images, device)
            with torch.no_grad():
                with context:
                    features = backbone.forward_features(backbone_inputs).tokens.to(device=device, dtype=fae_dtype)
                    z_tokens, _ = fae.encode(features)
            model_latents = bridge.to_model_latents(z_tokens)
            conditioning = _build_conditioning(backend, batch, device=device, class_conditioner=class_conditioner, text_conditioner=text_conditioner)

            optimizer.zero_grad(set_to_none=True)
            if conditioner_optimizer is not None:
                conditioner_optimizer.zero_grad(set_to_none=True)
            with context:
                out = backend(model_latents, conditioning=conditioning)
                bridge_cycle = bridge.cycle_loss(z_tokens) if bridge_cycle_weight > 0 else torch.zeros((), device=device)
                total = out.loss + bridge_cycle_weight * bridge_cycle
            if scaler.is_enabled():
                scaler.scale(total).backward()
                scaler.step(optimizer)
                if conditioner_optimizer is not None:
                    scaler.step(conditioner_optimizer)
                scaler.update()
            else:
                total.backward()
                optimizer.step()
                if conditioner_optimizer is not None:
                    conditioner_optimizer.step()
        except torch.cuda.OutOfMemoryError:
            if not skip_oom_batches:
                raise
            optimizer.zero_grad(set_to_none=True)
            if conditioner_optimizer is not None:
                conditioner_optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue
        except RuntimeError as exc:
            if not skip_oom_batches or 'out of memory' not in str(exc).lower():
                raise
            optimizer.zero_grad(set_to_none=True)
            if conditioner_optimizer is not None:
                conditioner_optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue

        running["loss"] += float(total.detach().cpu())
        running["backend_loss"] += out.logs.get("loss", float(out.loss.detach().cpu()))
        running["bridge_cycle"] += float(bridge_cycle.detach().cpu())
        count += 1
    return reduce_mean_dict(running, count=count, device=device)
