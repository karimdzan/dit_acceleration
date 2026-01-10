from dataclasses import dataclass
from pathlib import Path
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator
from tqdm import tqdm

from pfd_dit.backbones.sd15 import SD15Backbone


@dataclass
class TrainSD15Config:
    train_dir: Path
    out_dir: Path
    max_steps: int = 600
    batch_size: int = 4
    lr: float = 1e-6
    seed: int = 0
    mixed_precision: str = "fp16"
    grad_clip: float = 1.0
    gradient_checkpointing: bool = True


def finetune_sd15_unet(
    *,
    student_id: str,
    cfg: TrainSD15Config,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> Path:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)

    accelerator = Accelerator(mixed_precision=None if cfg.mixed_precision == "no" else cfg.mixed_precision)

    student = SD15Backbone.from_pretrained(student_id, device=device, torch_dtype=torch_dtype)
    unet = student.pipe.unet
    unet.train()

    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    student.pipe.vae.requires_grad_(False)
    student.pipe.text_encoder.requires_grad_(False)

    if cfg.mixed_precision == "fp16":
        unet.to(dtype=torch.float32)

    latents = torch.load(cfg.train_dir / "train_latents.pt", map_location="cpu")["latents"] # (N,4,64,64)
    embeds = torch.load(cfg.train_dir / "prompt_embeds.pt", map_location="cpu")["cond"] # (N,L,D)
    cond_ids = torch.load(cfg.train_dir / "prompt_ids.pt", map_location="cpu")["prompt_ids"] # (N,)

    ds = TensorDataset(latents, embeds, cond_ids)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)

    opt = torch.optim.AdamW(unet.parameters(), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=1e-2)

    unet, opt, dl = accelerator.prepare(unet, opt, dl)

    train_sched = student.train_scheduler
    T = int(train_sched.config.num_train_timesteps)
    pred_type = getattr(train_sched.config, "prediction_type", "epsilon")

    pbar = tqdm(total=cfg.max_steps, disable=not accelerator.is_local_main_process, desc="Finetune SD15 student")
    step = 0

    while step < cfg.max_steps:
        for x0, cemb, _cid in dl:
            if step >= cfg.max_steps:
                break

            param_dtype = next(unet.parameters()).dtype
            x0 = x0.to(device=accelerator.device, dtype=param_dtype)
            cemb = cemb.to(device=accelerator.device, dtype=param_dtype)

            bsz = x0.shape[0]
            t = torch.randint(0, T, (bsz,), device=accelerator.device, dtype=torch.long)

            noise = torch.randn_like(x0, dtype=torch.float32)
            noisy = train_sched.add_noise(x0.float(), noise, t).to(dtype=param_dtype)

            with accelerator.autocast():
                pred = unet(noisy, t, encoder_hidden_states=cemb).sample

                if pred_type == "epsilon":
                    target = noise
                elif pred_type == "v_prediction":
                    target = train_sched.get_velocity(x0.float(), noise, t)
                elif pred_type == "sample":
                    target = x0.float()
                else:
                    raise ValueError(f"Unsupported prediction_type={pred_type}")

                loss = F.mse_loss(pred.float(), target.float(), reduction="mean")

            if not torch.isfinite(loss):
                accelerator.print(f"[FATAL] step={step} loss non-finite: {loss}")
                raise FloatingPointError("Non-finite loss")

            accelerator.backward(loss)
            if cfg.grad_clip is not None:
                accelerator.clip_grad_norm_(unet.parameters(), cfg.grad_clip)

            opt.step()
            opt.zero_grad(set_to_none=True)

            if accelerator.is_local_main_process and step % 25 == 0:
                accelerator.print(f"step={step:6d} loss={loss.item():.6f}")

            step += 1
            pbar.update(1)

    pbar.close()

    save_dir = cfg.out_dir / "student_unet"
    save_dir.mkdir(parents=True, exist_ok=True)
    accelerator.unwrap_model(unet).save_pretrained(save_dir)

    (cfg.out_dir / "train_manifest.json").write_text(
        json.dumps({"student_id": student_id, "train_dir": str(cfg.train_dir)}, indent=2),
        encoding="utf-8",
    )

    return save_dir
