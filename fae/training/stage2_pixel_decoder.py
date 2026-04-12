from typing import Any

import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from fae.backbones.base import FrozenVisionBackbone
from fae.data.transforms import TargetImageTransform
from fae.models.discriminator import NLayerDiscriminator
from fae.models.fae import FeatureAutoEncoder
from fae.models.pixel_decoder import ViTPixelDecoder
from fae.training.common import prepare_backbone_inputs
from fae.utils.losses import VGGPerceptualLoss, discriminator_hinge_loss, image_reconstruction_loss


@torch.no_grad()
def build_stage2_preview(
    fae: FeatureAutoEncoder,
    pixel_decoder: ViTPixelDecoder,
    backbone: FrozenVisionBackbone,
    batch: dict[str, Any] | None,
    device: torch.device,
    image_size: int,
    mode: str = "gaussian",
    noise_std: float = 0.1,
    reference_decoder: ViTPixelDecoder | None = None,
) -> dict[str, torch.Tensor] | None:
    if batch is None:
        return None
    target_transform = TargetImageTransform(image_size)
    images = batch['images']
    targets = target_transform(images).to(device)
    backbone_inputs = prepare_backbone_inputs(backbone, images, device)
    features = backbone.forward_features(backbone_inputs).tokens.to(device=device, dtype=next(fae.parameters()).dtype)
    if mode == "gaussian":
        decoder_input = features + noise_std * torch.randn_like(features)
    elif mode == "finetune":
        decoder_input = fae(features).reconstructed_features
    else:
        raise ValueError(f"Unknown stage2 mode: {mode}")

    pred = pixel_decoder(decoder_input)
    out = {
        "target": targets.detach().cpu(),
        "pred": pred.detach().cpu(),
    }
    if reference_decoder is not None:
        reference_decoder.eval()
        ref_pred = reference_decoder(decoder_input)
        out["reference"] = ref_pred.detach().cpu()
    return out


def train_stage2_epoch(
    fae: FeatureAutoEncoder,
    pixel_decoder: ViTPixelDecoder,
    backbone: FrozenVisionBackbone,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    context,
    device: torch.device,
    image_size: int,
    mode: str = "gaussian",
    discriminator: NLayerDiscriminator | None = None,
    disc_optimizer: torch.optim.Optimizer | None = None,
    perceptual_loss: VGGPerceptualLoss | None = None,
    noise_std: float = 0.1,
    skip_oom_batches: bool = False,
) -> dict[str, float]:
    fae.eval()
    pixel_decoder.train()
    target_transform = TargetImageTransform(image_size)
    running = {"loss": 0.0, "recon": 0.0, "perceptual": 0.0, "adversarial": 0.0, "num_skipped": 0.0}
    count = 0
    for batch in tqdm(dataloader, desc="stage2", leave=False):
        if batch is None:
            running["num_skipped"] += 1
            continue
        images = batch['images']
        try:
            targets = target_transform(images).to(device)
            backbone_inputs = prepare_backbone_inputs(backbone, images, device)
            with torch.no_grad():
                features = backbone.forward_features(backbone_inputs).tokens.to(device=device, dtype=next(fae.parameters()).dtype)
                if mode == "gaussian":
                    decoder_input = features + noise_std * torch.randn_like(features)
                elif mode == "finetune":
                    decoder_input = fae(features).reconstructed_features
                else:
                    raise ValueError(f"Unknown stage2 mode: {mode}")

            disc_fake_logits = None
            with context:
                pred = pixel_decoder(decoder_input)
            if discriminator is not None and disc_optimizer is not None:
                discriminator.train()
                with torch.no_grad():
                    fake_for_disc = pred.detach()
                real_logits = discriminator(targets)
                fake_logits = discriminator(fake_for_disc)
                with context:
                    disc_loss = discriminator_hinge_loss(real_logits, fake_logits)
                disc_optimizer.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.scale(disc_loss).backward()
                    scaler.step(disc_optimizer)
                    scaler.update()
                else:
                    disc_loss.backward()
                    disc_optimizer.step()
                disc_fake_logits = discriminator(pred)

            with context:
                losses = image_reconstruction_loss(
                    pred,
                    targets,
                    perceptual_loss=perceptual_loss,
                    disc_fake_logits=disc_fake_logits,
                )
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(losses.total).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                losses.total.backward()
                optimizer.step()

            running['loss'] += float(losses.total.detach().cpu())
            running['recon'] += float(losses.reconstruction.detach().cpu())
            running['perceptual'] += float(losses.perceptual.detach().cpu())
            running['adversarial'] += float(losses.adversarial.detach().cpu())
            count += 1
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            if disc_optimizer is not None:
                disc_optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if skip_oom_batches:
                running['num_skipped'] += 1
                continue
            raise
        except Exception:
            optimizer.zero_grad(set_to_none=True)
            if disc_optimizer is not None:
                disc_optimizer.zero_grad(set_to_none=True)
            if skip_oom_batches:
                running['num_skipped'] += 1
                continue
            raise
    return {k: v / max(count, 1) if k != 'num_skipped' else v for k, v in running.items()}
