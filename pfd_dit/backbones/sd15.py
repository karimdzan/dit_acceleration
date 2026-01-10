from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDPMScheduler


@dataclass
class SD15Backbone:
    pipe: StableDiffusionPipeline
    train_scheduler: DDPMScheduler
    eval_scheduler: DDIMScheduler
    device: torch.device
    dtype: torch.dtype
    vae_scaling_factor: float
    latent_channels: int
    latent_size: int # 64 for 512px SD1.5

    @staticmethod
    def from_pretrained(
        model_id: str,
        device: torch.device,
        torch_dtype: torch.dtype = torch.float16,
    ) -> "SD15Backbone":
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(device)

        train_sched = DDPMScheduler.from_config(pipe.scheduler.config)
        eval_sched = DDIMScheduler.from_config(pipe.scheduler.config)

        vae_sf = float(getattr(pipe.vae.config, "scaling_factor", 0.18215))
        latent_channels = int(pipe.unet.config.in_channels)
        latent_size = int(getattr(pipe.unet.config, "sample_size", 64))

        return SD15Backbone(
            pipe=pipe,
            train_scheduler=train_sched,
            eval_scheduler=eval_sched,
            device=device,
            dtype=torch_dtype,
            vae_scaling_factor=vae_sf,
            latent_channels=latent_channels,
            latent_size=latent_size,
        )

    @torch.no_grad()
    def encode_prompts(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        tok = self.pipe.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.pipe.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(self.device)
        cond = self.pipe.text_encoder(tok.input_ids)[0]

        tok_u = self.pipe.tokenizer(
            [""] * len(prompts),
            padding="max_length",
            truncation=True,
            max_length=self.pipe.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(self.device)
        uncond = self.pipe.text_encoder(tok_u.input_ids)[0]
        return cond, uncond

    @torch.no_grad()
    def decode_latents_to_images_01(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.to(self.device)
        imgs = self.pipe.vae.decode(latents / self.vae_scaling_factor).sample
        return (imgs.clamp(-1, 1) + 1) / 2

    @torch.no_grad()
    def sample_latents_and_images(
        self,
        *,
        cond_uncond_embeds: Tuple[torch.Tensor, torch.Tensor],
        init_latents: Optional[torch.Tensor],
        num_inference_steps: int,
        guidance_scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cond, uncond = cond_uncond_embeds
        bsz = cond.shape[0]

        sched = self.eval_scheduler
        sched.set_timesteps(num_inference_steps, device=self.device)
        timesteps = sched.timesteps

        if init_latents is None:
            latents = torch.randn(
                (bsz, self.latent_channels, self.latent_size, self.latent_size),
                device=self.device,
                dtype=self.dtype,
            )
            latents = latents * float(getattr(sched, "init_noise_sigma", 1.0))
        else:
            latents = init_latents.to(self.device, dtype=self.dtype)

        unet = self.pipe.unet
        unet.eval()

        for t in timesteps:
            t_vec = torch.full((bsz,), int(t.item()), device=self.device, dtype=torch.long)

            if guidance_scale is not None and float(guidance_scale) > 1.0:
                lat_in = torch.cat([latents, latents], dim=0)
                emb_in = torch.cat([uncond, cond], dim=0)
                t_in = torch.cat([t_vec, t_vec], dim=0)

                eps = unet(lat_in, t_in, encoder_hidden_states=emb_in).sample
                eps_u, eps_c = eps.chunk(2, dim=0)
                eps = eps_u + float(guidance_scale) * (eps_c - eps_u)
            else:
                eps = unet(latents, t_vec, encoder_hidden_states=cond).sample

            latents = sched.step(eps, t, latents).prev_sample

        imgs = self.decode_latents_to_images_01(latents)
        return latents, imgs
