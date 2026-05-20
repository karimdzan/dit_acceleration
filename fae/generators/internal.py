import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fae.modules.rmsnorm import RMSNorm
from fae.modules.transformer import SelfAttention, SinusoidalTimestepEmbedding

from .base import LatentGeneratorBackend
from .common import ConditioningBundle, LatentTensorSpec, LossOutput
from .objectives import FlowMatchingObjective, LinearVelocityTransportObjective, SimpleCosineDiffusionObjective


def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    if embed_dim % 4 != 0:
        raise ValueError(f"2D sin-cos embedding requires embed_dim divisible by 4, got {embed_dim}.")
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
    grid = torch.stack(grid, dim=0).reshape(2, 1, grid_size, grid_size)

    def _get_1d_pos_embed(dim: int, pos: torch.Tensor) -> torch.Tensor:
        omega = torch.arange(dim // 2, dtype=torch.float32)
        omega = 1.0 / (10000 ** (omega / max(dim // 2, 1)))
        out = pos.reshape(-1, 1) * omega.reshape(1, -1)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    emb_h = _get_1d_pos_embed(embed_dim // 2, grid[0])
    emb_w = _get_1d_pos_embed(embed_dim // 2, grid[1])
    return torch.cat([emb_h, emb_w], dim=1)


def _modulate(x: torch.Tensor, shift: torch.Tensor | None, scale: torch.Tensor) -> torch.Tensor:
    if shift is None:
        return x * (1 + scale)
    return x * (1 + scale) + shift


def _gate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return x * gate


class _SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) :
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim)
        self.value = nn.Linear(dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(F.silu(self.gate(x)) * self.value(x))


class _NoAffineNorm(nn.Module):
    def __init__(self, dim: int, use_rmsnorm: bool = True) :
        super().__init__()
        self.norm = RMSNorm(dim, eps=1e-6) if use_rmsnorm else nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class _DDTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int,
        mlp_ratio: float = 4.0,
        qk_head_dim: int | None = None,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        use_swiglu: bool = True,
        wo_shift: bool = False,
    ) :
        super().__init__()
        self.norm1 = _NoAffineNorm(dim, use_rmsnorm=use_rmsnorm)
        self.norm2 = _NoAffineNorm(dim, use_rmsnorm=use_rmsnorm)
        self.attn = SelfAttention(dim, num_heads=num_heads, head_dim=qk_head_dim, use_rope_2d=use_rope)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = _SwiGLUFFN(dim, int((2.0 / 3.0) * mlp_hidden_dim)) if use_swiglu else nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, dim),
        )
        self.wo_shift = bool(wo_shift)
        mod_dim = 4 * dim if self.wo_shift else 6 * dim
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, mod_dim, bias=True))

    def forward(self, x: torch.Tensor, cond: torch.Tensor, grid_size: tuple[int, int]) -> torch.Tensor:
        if cond.ndim == 2:
            cond = cond.unsqueeze(1)
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(4, dim=-1)
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=-1)
        x = x + _gate(self.attn(_modulate(self.norm1(x), shift_msa, scale_msa), grid_size=grid_size), gate_msa)
        x = x + _gate(self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp)), gate_mlp)
        return x


class _DDTFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, cond_dim: int, use_rmsnorm: bool = True) :
        super().__init__()
        self.norm_final = _NoAffineNorm(hidden_size, use_rmsnorm=use_rmsnorm)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * hidden_size, bias=True))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if cond.ndim == 2:
            cond = cond.unsqueeze(1)
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1)
        return self.linear(_modulate(self.norm_final(x), shift, scale))


class _InternalLatentNetwork(nn.Module):
    def __init__(
        self,
        spec: LatentTensorSpec,
        model_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        cond_dim: int = 1024,
        mlp_ratio: float = 4.0,
        head_dim: int | None = None,
        use_rope_2d: bool = False,
    ) :
        super().__init__()
        self.spec = spec
        self.in_proj = nn.Linear(spec.channels, model_dim)
        self.pos = nn.Parameter(torch.zeros(1, spec.height * spec.width, model_dim))
        self.time_embed = SinusoidalTimestepEmbedding(cond_dim)
        self.blocks = nn.ModuleList([
            _DDTBlock(
                model_dim,
                num_heads=num_heads,
                cond_dim=cond_dim,
                mlp_ratio=mlp_ratio,
                qk_head_dim=head_dim,
                use_rope=use_rope_2d,
                use_rmsnorm=True,
                use_swiglu=True,
                wo_shift=False,
            )
            for _ in range(depth)
        ])
        self.norm = RMSNorm(model_dim, eps=1e-6)
        self.out_proj = nn.Linear(model_dim, spec.channels)
        self.default_cond = nn.Parameter(torch.zeros(cond_dim))
        self._initialize_weights()

    def _initialize_weights(self) :
        nn.init.normal_(self.time_embed.proj[0].weight, std=0.02)
        nn.init.zeros_(self.time_embed.proj[0].bias)
        nn.init.normal_(self.time_embed.proj[2].weight, std=0.02)
        nn.init.zeros_(self.time_embed.proj[2].bias)
        nn.init.normal_(self.pos, std=0.02)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, conditioning: ConditioningBundle | None = None) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.reshape(b, c, h * w).transpose(1, 2)
        cond = self.time_embed(t)
        if conditioning is not None and conditioning.vector is not None:
            cond = cond + conditioning.vector
        else:
            cond = cond + self.default_cond.unsqueeze(0)
        y = self.in_proj(tokens) + self.pos
        for block in self.blocks:
            y = block(y, cond, grid_size=(h, w))
        y = self.out_proj(self.norm(y))
        return y.transpose(1, 2).reshape(b, c, h, w)


class _DiTDHNetwork(nn.Module):
    def __init__(
        self,
        spec: LatentTensorSpec,
        trunk_dim: int = 1152,
        head_dim: int = 2048,
        trunk_depth: int = 28,
        head_depth: int = 2,
        num_heads_trunk: int = 16,
        num_heads_head: int = 16,
        num_classes: int = 1000,
        class_dropout_prob: float = 0.1,
        mlp_ratio: float = 4.0,
        qk_head_dim: int | None = None,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        use_swiglu: bool = True,
        use_pos_embed: bool = True,
        wo_shift: bool = False,
    ) :
        super().__init__()
        self.spec = spec
        self.num_classes = int(num_classes)
        self.class_dropout_prob = float(class_dropout_prob)
        self.use_pos_embed = bool(use_pos_embed)

        # Follow the released DiT^DH layout more closely:
        # latent -> structural stream s (trunk/encoder blocks) and latent -> x (decoder/head blocks).
        self.s_embedder = nn.Linear(spec.channels, trunk_dim)
        self.x_embedder = nn.Linear(spec.channels, head_dim)
        self.s_projector = nn.Linear(trunk_dim, head_dim) if trunk_dim != head_dim else nn.Identity()
        self.t_embedder = SinusoidalTimestepEmbedding(trunk_dim)
        self.y_embedder = nn.Embedding(self.num_classes + 1, trunk_dim)
        self.null_class_id = self.num_classes

        if self.use_pos_embed:
            pos = _get_2d_sincos_pos_embed(trunk_dim, spec.height).unsqueeze(0)
            self.register_buffer("pos_embed", pos, persistent=True)
        else:
            self.pos_embed = None

        self.encoder_blocks = nn.ModuleList([
            _DDTBlock(
                trunk_dim,
                num_heads=num_heads_trunk,
                cond_dim=trunk_dim,
                mlp_ratio=mlp_ratio,
                qk_head_dim=qk_head_dim,
                use_rope=use_rope,
                use_rmsnorm=use_rmsnorm,
                use_swiglu=use_swiglu,
                wo_shift=wo_shift,
            )
            for _ in range(trunk_depth)
        ])
        self.decoder_blocks = nn.ModuleList([
            _DDTBlock(
                head_dim,
                num_heads=num_heads_head,
                cond_dim=head_dim,
                mlp_ratio=mlp_ratio,
                qk_head_dim=qk_head_dim,
                use_rope=use_rope,
                use_rmsnorm=use_rmsnorm,
                use_swiglu=use_swiglu,
                wo_shift=wo_shift,
            )
            for _ in range(head_depth)
        ])
        self.final_layer = _DDTFinalLayer(head_dim, spec.channels, cond_dim=head_dim, use_rmsnorm=use_rmsnorm)
        self.initialize_weights()

    def initialize_weights(self) :
        for linear in (self.s_embedder, self.x_embedder):
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
        if isinstance(self.s_projector, nn.Linear):
            nn.init.xavier_uniform_(self.s_projector.weight)
            nn.init.zeros_(self.s_projector.bias)

        nn.init.normal_(self.y_embedder.weight, std=0.02)
        nn.init.normal_(self.t_embedder.proj[0].weight, std=0.02)
        nn.init.zeros_(self.t_embedder.proj[0].bias)
        nn.init.normal_(self.t_embedder.proj[2].weight, std=0.02)
        nn.init.zeros_(self.t_embedder.proj[2].bias)

        for block in list(self.encoder_blocks) + list(self.decoder_blocks):
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def _prepare_class_labels(self, conditioning: ConditioningBundle | None, batch_size: int, device: torch.device) -> torch.Tensor:
        if conditioning is None or conditioning.class_labels is None:
            labels = torch.full((batch_size,), self.null_class_id, device=device, dtype=torch.long)
        else:
            labels = conditioning.class_labels.to(device=device, dtype=torch.long)
            if self.training and self.class_dropout_prob > 0:
                drop_mask = torch.rand(batch_size, device=device) < self.class_dropout_prob
                labels = labels.clone()
                labels[drop_mask] = self.null_class_id
        return labels

    def forward(self, x: torch.Tensor, t: torch.Tensor, conditioning: ConditioningBundle | None = None) -> torch.Tensor:
        b, c, h, w = x.shape
        if (c, h, w) != (self.spec.channels, self.spec.height, self.spec.width):
            raise ValueError(
                f"DiTDH expected latent shape {(self.spec.channels, self.spec.height, self.spec.width)}, got {(c, h, w)}"
            )

        tokens = x.reshape(b, c, h * w).transpose(1, 2)
        labels = self._prepare_class_labels(conditioning, b, x.device)
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(labels)
        cond = F.silu(t_emb + y_emb)

        s = self.s_embedder(tokens)
        if self.pos_embed is not None:
            s = s + self.pos_embed.to(device=s.device, dtype=s.dtype)
        for block in self.encoder_blocks:
            s = block(s, cond, grid_size=(h, w))

        s = F.silu(t_emb.unsqueeze(1) + s)
        s = self.s_projector(s)

        x_tokens = self.x_embedder(tokens)
        for block in self.decoder_blocks:
            x_tokens = block(x_tokens, s, grid_size=(h, w))

        out = self.final_layer(x_tokens, s)
        return out.transpose(1, 2).reshape(b, c, h, w)


class InternalLatentDiTBackend(LatentGeneratorBackend):
    def __init__(
        self,
        spec: LatentTensorSpec,
        model_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        cond_dim: int = 1024,
        mlp_ratio: float = 4.0,
        objective: str = "diffusion",
        prediction_type: str = "v_prediction",
        time_shift: float = 0.0,
        head_dim: int | None = None,
        use_rope_2d: bool = False,
    ) :
        super().__init__()
        self.spec = spec
        self.model = _InternalLatentNetwork(
            spec,
            model_dim=model_dim,
            depth=depth,
            num_heads=num_heads,
            cond_dim=cond_dim,
            mlp_ratio=mlp_ratio,
            head_dim=head_dim,
            use_rope_2d=use_rope_2d,
        )
        self.objective = FlowMatchingObjective() if objective == "flow_matching" else SimpleCosineDiffusionObjective(prediction_type=prediction_type)
        self.time_shift = time_shift

    def latent_spec(self) -> LatentTensorSpec:
        return self.spec

    def training_loss(self, latents: torch.Tensor, conditioning: ConditioningBundle | None = None) -> LossOutput:
        if isinstance(self.objective, SimpleCosineDiffusionObjective):
            return self.objective.training_loss(self.model, latents, conditioning, time_shift=self.time_shift)
        return self.objective.training_loss(self.model, latents, conditioning)

    def sample_latents(self, batch_size: int, device: torch.device, conditioning: ConditioningBundle | None = None, num_steps: int = 30, guidance_scale: float = 1.0) -> torch.Tensor:
        shape = (batch_size, self.spec.channels, self.spec.height, self.spec.width)
        if isinstance(self.objective, SimpleCosineDiffusionObjective):
            return self.objective.sample(self.model, shape, device, conditioning=conditioning, num_steps=num_steps, time_shift=self.time_shift)
        return self.objective.sample(self.model, shape, device, conditioning=conditioning, num_steps=num_steps)


class InternalDiTDHBackend(LatentGeneratorBackend):
    uses_raw_class_labels = True

    def __init__(
        self,
        spec: LatentTensorSpec,
        hidden_size: tuple[int, int] = (1152, 2048),
        depth: tuple[int, int] = (28, 2),
        num_heads: tuple[int, int] = (16, 16),
        mlp_ratio: float = 4.0,
        num_classes: int = 1000,
        class_dropout_prob: float = 0.1,
        use_qknorm: bool = False,
        use_swiglu: bool = True,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        wo_shift: bool = False,
        use_pos_embed: bool = True,
        objective: str = "linear_velocity",
        prediction_type: str = "velocity",
        time_dist_type: str = "logit-normal_0_1",
        loss_weight: str | None = None,
    ) :
        super().__init__()
        del use_qknorm, prediction_type
        self.spec = spec
        self.model = _DiTDHNetwork(
            spec=spec,
            trunk_dim=int(hidden_size[0]),
            head_dim=int(hidden_size[1]),
            trunk_depth=int(depth[0]),
            head_depth=int(depth[1]),
            num_heads_trunk=int(num_heads[0]),
            num_heads_head=int(num_heads[1]),
            num_classes=int(num_classes),
            class_dropout_prob=float(class_dropout_prob),
            mlp_ratio=float(mlp_ratio),
            use_rope=bool(use_rope),
            use_rmsnorm=bool(use_rmsnorm),
            use_swiglu=bool(use_swiglu),
            use_pos_embed=bool(use_pos_embed),
            wo_shift=bool(wo_shift),
        )
        objective = objective.lower()
        if objective not in {"linear_velocity", "velocity", "flow_matching"}:
            raise ValueError(f"Unsupported InternalDiTDH objective={objective}")
        self.objective = LinearVelocityTransportObjective(time_dist_type=time_dist_type, loss_weight=loss_weight)

    def latent_spec(self) -> LatentTensorSpec:
        return self.spec

    def _guided_model(self, x: torch.Tensor, t: torch.Tensor, conditioning: ConditioningBundle | None, guidance_scale: float) -> torch.Tensor:
        if guidance_scale == 1.0 or conditioning is None or conditioning.class_labels is None:
            return self.model(x, t, conditioning)
        cond_pred = self.model(x, t, conditioning)
        null_bundle = ConditioningBundle(class_labels=torch.full_like(conditioning.class_labels, self.model.null_class_id))
        uncond_pred = self.model(x, t, null_bundle)
        return uncond_pred + guidance_scale * (cond_pred - uncond_pred)

    def training_loss(self, latents: torch.Tensor, conditioning: ConditioningBundle | None = None) -> LossOutput:
        return self.objective.training_loss(self.model, latents, conditioning)

    def sample_latents(self, batch_size: int, device: torch.device, conditioning: ConditioningBundle | None = None, num_steps: int = 50, guidance_scale: float = 1.0) -> torch.Tensor:
        shape = (batch_size, self.spec.channels, self.spec.height, self.spec.width)
        x = torch.randn(shape, device=device)
        ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        for i in range(num_steps):
            t = torch.full((batch_size,), ts[i].item(), device=device)
            dt = ts[i + 1] - ts[i]
            v = self._guided_model(x, t, conditioning, guidance_scale)
            x = x + dt * v
        return x
