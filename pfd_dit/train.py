from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator

from .data import LatentsDataset
from .utils.seed import seed_everything

@dataclass
class TrainConfig:
    out_dir: Path
    max_train_steps: int = 600
    train_batch_size: int = 8
    grad_accum_steps: int = 1
    lr: float = 5e-5
    weight_decay: float = 0.0
    mixed_precision: str = "bf16"  # "no"|"fp16"|"bf16"
    seed: int = 0
    log_every: int = 50
    save_every: int = 200

def finetune_student_on_latents(
    *,
    student_pipe,
    train_latents: torch.Tensor,
    train_labels: torch.Tensor,
    cfg: TrainConfig,
) -> Path:
    '''
    Fine-tune the student DiT on a limited dataset to induce memorization/generalization effects
    without any full pretraining.
    '''
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)

    accelerator = Accelerator(
        mixed_precision=None if cfg.mixed_precision == "no" else cfg.mixed_precision,
        gradient_accumulation_steps=cfg.grad_accum_steps,
    )
    device = accelerator.device

    transformer = student_pipe.transformer
    transformer.train()

    if cfg.mixed_precision == "no":
        transformer.to(dtype=torch.float32)

    ds = LatentsDataset(train_latents, train_labels)
    dl = DataLoader(ds, batch_size=cfg.train_batch_size, shuffle=True, drop_last=True)

    opt = torch.optim.AdamW(transformer.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    transformer, opt, dl = accelerator.prepare(transformer, opt, dl)

    train_scheduler = getattr(student_pipe, "_train_scheduler", student_pipe.scheduler)
    num_train_timesteps = int(getattr(train_scheduler.config, "num_train_timesteps", 1000))
    prediction_type = getattr(train_scheduler.config, "prediction_type", "epsilon")
    
    global_step = 0
    pbar = tqdm(total=cfg.max_train_steps, desc="Finetuning student", disable=not accelerator.is_local_main_process)

    while global_step < cfg.max_train_steps:
        for latents, labels in dl:
            if global_step >= cfg.max_train_steps:
                break

            labels = labels.to(device)

            bsz = latents.shape[0]
            timesteps = torch.randint(0, num_train_timesteps, (bsz,), device=device, dtype=torch.long)
            param_dtype = next(transformer.parameters()).dtype
            latents = latents.to(device=device, dtype=param_dtype)
            noise = torch.randn_like(latents)
            noise = noise.to(device=device, dtype=param_dtype)
            noisy = train_scheduler.add_noise(latents.float(), noise.float(), timesteps).to(latents.dtype)
            noisy = noisy.to(device=device, dtype=param_dtype)

            with accelerator.accumulate(transformer):
                out = transformer(noisy, timesteps, class_labels=labels)
                model_out = out.sample

                
                c_lat = noisy.shape[1]
                if model_out.shape[1] == 2 * c_lat:
                    model_pred, _ = torch.chunk(model_out, 2, dim=1)
                else:
                    model_pred = model_out
                    if model_pred.shape[1] != c_lat:
                        model_pred = model_pred[:, :c_lat, :, :]
                
                if prediction_type == "epsilon":
                    target = noise
                elif prediction_type == "v_prediction":
                    target = train_scheduler.get_velocity(latents.float(), noise.float(), timesteps)
                elif prediction_type == "sample":
                    target = latents.float()
                else:
                    raise ValueError(f"Unsupported prediction_type: {prediction_type}")

                target = target.to(model_pred.dtype)

                loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")

                bad = False
                if not torch.isfinite(model_pred).all():
                    bad = True
                    reason = "model_pred non-finite"
                elif not torch.isfinite(loss):
                    bad = True
                    reason = "loss non-finite"

                if bad:
                    max_lat = float(latents.detach().abs().max().item())
                    max_noi = float(noise.detach().abs().max().item())
                    max_nsy = float(noisy.detach().abs().max().item())
                    max_out = float(model_pred.detach().abs().max().item()) if "model_pred" in locals() else float("nan")

                    accelerator.print(
                        f"[FATAL] step={global_step} {reason} | "
                        f"max|lat|={max_lat:.3e} max|noise|={max_noi:.3e} max|noisy|={max_nsy:.3e} max|pred|={max_out:.3e}"
                    )
                    raise FloatingPointError(reason)

                accelerator.backward(loss)

                accelerator.clip_grad_norm_(transformer.parameters(), 1.0)


                if accelerator.is_local_main_process and global_step < 5:
                    gmax = 0.0
                    for p in transformer.parameters():
                        if p.grad is None:
                            continue
                        if not torch.isfinite(p.grad).all():
                            accelerator.print("Found non-finite grad BEFORE step()")
                            break
                        gmax = max(gmax, float(p.grad.detach().abs().max().item()))
                    accelerator.print(f"step={global_step} loss={loss.item():.6f} grad_max={gmax:.3e}")

                opt.step()

                if accelerator.is_local_main_process and global_step < 5:
                    pmax = 0.0
                    for p in transformer.parameters():
                        if not torch.isfinite(p).all():
                            accelerator.print("Found non-finite PARAM AFTER step()")
                            break
                        pmax = max(pmax, float(p.detach().abs().max().item()))
                    accelerator.print(f"step={global_step} param_max={pmax:.3e}")

                opt.zero_grad(set_to_none=True)

            if accelerator.is_local_main_process and (global_step % cfg.log_every == 0):
                accelerator.print(f"step={global_step:6d} loss={loss.item():.6f}")

            if accelerator.is_local_main_process and cfg.save_every > 0 and (global_step > 0) and (global_step % cfg.save_every == 0):
                save_dir = cfg.out_dir / f"student_step{global_step}"
                unwrapped = accelerator.unwrap_model(transformer)
                student_pipe.transformer = unwrapped
                student_pipe.save_pretrained(save_dir)

            global_step += 1
            pbar.update(1)

    pbar.close()

    if accelerator.is_local_main_process:
        unwrapped = accelerator.unwrap_model(transformer)
        student_pipe.transformer = unwrapped
        student_pipe.save_pretrained(cfg.out_dir / "student")

    accelerator.wait_for_everyone()
    return cfg.out_dir / "student"
