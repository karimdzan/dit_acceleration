from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fae.backbones.base import FrozenVisionBackbone
from fae.models.discriminator import NLayerDiscriminator
from fae.models.rae import RepresentationAutoEncoder
from fae.training.common import clip_grad_norm_, get_current_lr, prepare_backbone_inputs
from fae.utils.losses import VGGPerceptualLoss, discriminator_hinge_loss, image_reconstruction_loss


def _adaptive_gan_weight(reconstruction_loss: torch.Tensor, adversarial_loss: torch.Tensor, last_layer: torch.Tensor, base_weight: float) -> torch.Tensor:
    rec_grad = torch.autograd.grad(reconstruction_loss, last_layer, retain_graph=True, allow_unused=True)[0]
    adv_grad = torch.autograd.grad(adversarial_loss, last_layer, retain_graph=True, allow_unused=True)[0]
    if rec_grad is None or adv_grad is None:
        return reconstruction_loss.new_tensor(float(base_weight))
    ratio = rec_grad.norm() / adv_grad.norm().clamp(min=1e-4)
    return ratio.detach().clamp(0.0, 1e4) * float(base_weight)


@torch.no_grad()
def build_stage1_preview(
    model: RepresentationAutoEncoder,
    backbone: FrozenVisionBackbone,
    batch: dict[str, Any] | None,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    if batch is None:
        return None
    images = batch["images"]
    targets = backbone.build_reconstruction_targets(images, output_size=model.image_size).to(device)
    backbone_inputs = prepare_backbone_inputs(backbone, images, device)
    features = backbone.forward_features(backbone_inputs).tokens.to(device=device, dtype=next(model.parameters()).dtype)
    pred = model(features, add_noise=False).reconstructed_images
    return {"target": targets.detach().cpu(), "pred": pred.detach().cpu()}


def train_stage1_epoch(
    model: RepresentationAutoEncoder,
    backbone: FrozenVisionBackbone,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    context,
    device: torch.device,
    discriminator: NLayerDiscriminator | None = None,
    disc_optimizer: torch.optim.Optimizer | None = None,
    perceptual_loss: VGGPerceptualLoss | None = None,
    current_epoch: int = 0,
    perceptual_weight: float = 0.1,
    adversarial_weight: float = 0.01,
    use_adaptive_gan_weight: bool = True,
    lpips_start_epoch: int = 0,
    disc_start_epoch: int = 6,
    adv_start_epoch: int = 8,
    skip_oom_batches: bool = True,
    accumulation_steps: int = 1,
    grad_clip: float | None = None,
    lr_scheduler=None,
    ema=None,
) -> dict[str, float]:
    model.train()
    backbone.eval()
    if discriminator is not None:
        discriminator.train()
    running = {
        "loss": 0.0,
        "recon": 0.0,
        "perceptual": 0.0,
        "adversarial": 0.0,
        "disc": 0.0,
        "disc_weight": 0.0,
        "grad_norm": 0.0,
        "num_skipped": 0.0,
    }
    count = 0

    accumulation_steps = max(int(accumulation_steps), 1)
    use_lpips = perceptual_loss is not None and current_epoch >= lpips_start_epoch
    use_disc = discriminator is not None and disc_optimizer is not None and current_epoch >= disc_start_epoch
    use_adv = use_disc and current_epoch >= adv_start_epoch and adversarial_weight > 0
    optimizer.zero_grad(set_to_none=True)
    if disc_optimizer is not None:
        disc_optimizer.zero_grad(set_to_none=True)

    for step_idx, batch in enumerate(tqdm(dataloader, desc="stage1", leave=False)):
        if batch is None:
            running["num_skipped"] += 1
            continue
        images = batch["images"]
        try:
            targets = backbone.build_reconstruction_targets(images, output_size=model.image_size).to(device)
            backbone_inputs = prepare_backbone_inputs(backbone, images, device)

            with torch.no_grad():
                with context:
                    features = backbone.forward_features(backbone_inputs).tokens.to(
                        device=device,
                        dtype=next(model.parameters()).dtype,
                    )
            with context:
                out = model(features, add_noise=True)
                pred = out.reconstructed_images

            disc_fake_logits_for_gen = None
            if use_disc and discriminator is not None and disc_optimizer is not None:
                real_logits = discriminator(targets)
                fake_logits = discriminator(pred.detach())
                with context:
                    disc_loss = discriminator_hinge_loss(real_logits, fake_logits) / accumulation_steps
                if scaler.is_enabled():
                    scaler.scale(disc_loss).backward()
                else:
                    disc_loss.backward()
                if ((step_idx + 1) % accumulation_steps == 0) or (step_idx + 1 == len(dataloader)):
                    if scaler.is_enabled():
                        scaler.step(disc_optimizer)
                    else:
                        disc_optimizer.step()
                    disc_optimizer.zero_grad(set_to_none=True)
                if use_adv:
                    disc_fake_logits_for_gen = discriminator(pred)
            else:
                disc_loss = torch.zeros((), device=device, dtype=pred.dtype)

            perceptual_module = perceptual_loss if use_lpips else None
            with context:
                losses = image_reconstruction_loss(
                    pred,
                    targets,
                    perceptual_loss=perceptual_module,
                    disc_fake_logits=None,
                    adversarial_weight=1.0,
                    perceptual_weight=perceptual_weight,
                )
                adv_loss = pred.new_zeros(())
                disc_weight = pred.new_zeros(())
                total = losses.reconstruction + losses.perceptual
                if disc_fake_logits_for_gen is not None and use_adv:
                    adv_loss = -disc_fake_logits_for_gen.mean()
                    if use_adaptive_gan_weight:
                        disc_weight = _adaptive_gan_weight(total, adv_loss, model.decoder.patch_head.weight, adversarial_weight)
                    else:
                        disc_weight = pred.new_tensor(float(adversarial_weight))
                    total = total + disc_weight * adv_loss
                total = total / accumulation_steps

            if scaler.is_enabled():
                scaler.scale(total).backward()
            else:
                total.backward()

            grad_norm = 0.0
            if ((step_idx + 1) % accumulation_steps == 0) or (step_idx + 1 == len(dataloader)):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                grad_norm = clip_grad_norm_(model.decoder.parameters(), grad_clip)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if lr_scheduler is not None:
                    lr_scheduler.step()
                if ema is not None:
                    ema.update(model)

            running["loss"] += float((total * accumulation_steps).detach().cpu())
            running["recon"] += float(losses.reconstruction.detach().cpu())
            running["perceptual"] += float(losses.perceptual.detach().cpu())
            running["adversarial"] += float(adv_loss.detach().cpu())
            running["disc"] += float((disc_loss * accumulation_steps).detach().cpu())
            running["disc_weight"] += float(disc_weight.detach().cpu())
            running["grad_norm"] += float(grad_norm)
            count += 1
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            if disc_optimizer is not None:
                disc_optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if skip_oom_batches:
                running["num_skipped"] += 1
                continue
            raise
        except RuntimeError as exc:
            optimizer.zero_grad(set_to_none=True)
            if disc_optimizer is not None:
                disc_optimizer.zero_grad(set_to_none=True)
            if skip_oom_batches and "out of memory" in str(exc).lower():
                running["num_skipped"] += 1
                torch.cuda.empty_cache()
                continue
            raise

    averaged = {k: (v / max(count, 1) if k != "num_skipped" else v) for k, v in running.items()}
    averaged["lr"] = get_current_lr(optimizer)
    return averaged
