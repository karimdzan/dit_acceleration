import argparse
import time
from dataclasses import asdict

import torch
import torch.nn as nn

from fae.config import load_yaml
from fae.generators.common import ConditioningBundle, LatentTensorSpec
from fae.generators.registry import build_generator_backend
from fae.modules.token_merging import apply_token_merging, iter_token_merging_wrappers


class _RecordingBlock(nn.Module):
    def __init__(self, dim: int) :
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.seen_tokens: int | None = None
        self.seen_grid_size: tuple[int, int] | None = None

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None, grid_size: tuple[int, int] | None = None) -> torch.Tensor:
        self.seen_tokens = int(x.shape[1])
        self.seen_grid_size = grid_size
        return x + self.proj(self.norm(x))


class _FakeDiTXL(nn.Module):
    """Small DiT-XL-shaped block container used for CPU validation."""

    def __init__(self, depth: int = 28, dim: int = 64) :
        super().__init__()
        self.blocks = nn.ModuleList([_RecordingBlock(dim) for _ in range(depth)])

    def forward(self, x: torch.Tensor, grid_size: tuple[int, int]) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, None, grid_size=grid_size)
        return x


def _sync(device: torch.device) :
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_call(fn, warmup: int, repeats: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    _sync(device)
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    _sync(device)
    return (time.perf_counter() - start) / max(1, repeats)


def _print_wrapper_stats(model: nn.Module) :
    wrappers = list(iter_token_merging_wrappers(model))
    print(f"wrapped_layers={getattr(model, 'token_merging_layers', 'unknown')}")
    print(f"num_wrappers={len(wrappers)}")
    for i, wrapper in enumerate(wrappers[:8]):
        stats = wrapper.last_stats
        if stats is None:
            print(f"wrapper[{i}]: not called yet")
        else:
            print(f"wrapper[{i}]: {asdict(stats)} keep_ratio={stats.keep_ratio:.3f}")
    if len(wrappers) > 8:
        print(f"... {len(wrappers) - 8} more wrappers omitted")


def run_dry_run(args: argparse.Namespace) :
    device = torch.device(args.device)
    cfg = {
        "enabled": True,
        "method": "spatial_pool",
        "stride": args.stride,
        "start_layer": args.start_layer,
        "end_layer": args.end_layer,
        "every": args.every,
        "num_prefix_tokens": 0,
        "min_image_tokens": 64,
    }
    model = _FakeDiTXL(depth=args.depth, dim=args.dim).to(device)
    wrapped = apply_spatial_token_merging(model, cfg)

    h = w = args.grid_size
    x = torch.randn(args.batch_size, h * w, args.dim, device=device)
    with torch.no_grad():
        y = model(x, grid_size=(h, w))

    assert y.shape == x.shape, f"shape mismatch: input={tuple(x.shape)} output={tuple(y.shape)}"
    _print_wrapper_stats(model)
    print(f"dry_run_ok=true input_shape={tuple(x.shape)} output_shape={tuple(y.shape)} wrapped={wrapped}")

    if args.benchmark:
        dense = _FakeDiTXL(depth=args.depth, dim=args.dim).to(device).eval()
        merged = model.eval()
        with torch.no_grad():
            dense_s = _time_call(lambda: dense(x, grid_size=(h, w)), args.warmup, args.repeats, device)
            merged_s = _time_call(lambda: merged(x, grid_size=(h, w)), args.warmup, args.repeats, device)
        print(f"dense_seconds={dense_s:.6f}")
        print(f"merged_seconds={merged_s:.6f}")
        if merged_s > 0:
            print(f"speedup={dense_s / merged_s:.3f}x")


def run_config_validation(args: argparse.Namespace) :
    config = load_yaml(args.config)
    gen_cfg = config.setdefault("generator", {})
    token_merging_cfg = gen_cfg.setdefault("token_merging", {})
    token_merging_cfg.setdefault("enabled", True)
    token_merging_cfg.setdefault("method", "spatial_pool")
    token_merging_cfg.setdefault("stride", args.stride)
    token_merging_cfg.setdefault("start_layer", args.start_layer)
    token_merging_cfg.setdefault("end_layer", args.end_layer)
    token_merging_cfg.setdefault("every", args.every)
    token_merging_cfg.setdefault("num_prefix_tokens", 0)

    model_spec = LatentTensorSpec(
        channels=int(gen_cfg.get("in_channels", 4)),
        height=int(gen_cfg.get("sample_size", args.grid_size)),
        width=int(gen_cfg.get("sample_size", args.grid_size)),
    )
    backend = build_generator_backend(config, bridge_spec=model_spec)
    device = torch.device(args.device)
    backend.to(device)
    backend.eval()

    spec = backend.latent_spec()
    dtype = next(backend.model.parameters()).dtype
    latents = torch.randn(args.batch_size, spec.channels, spec.height, spec.width, device=device, dtype=dtype)

    if gen_cfg.get("name") == "diffusers_dit":
        labels = torch.zeros(args.batch_size, device=device, dtype=torch.long)
        conditioning = ConditioningBundle(class_labels=labels)
    else:
        conditioning = None

    with torch.no_grad():
        out = backend.training_loss(latents, conditioning=conditioning)
    print(f"loss={float(out.loss.detach().cpu()):.6f}")
    _print_wrapper_stats(backend.model)
    print(f"config_validation_ok=true latent_shape={tuple(latents.shape)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None, help="Stage-3 config to validate with a real backend.")
    parser.add_argument("--dry-run", action="store_true", help="Use a fake DiT-XL-shaped model; does not require diffusers/checkpoints.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grid-size", type=int, default=32, help="Latent grid side. DiT-XL/2-256 uses 32.")
    parser.add_argument("--depth", type=int, default=28, help="Fake model depth. DiT-XL has 28 transformer blocks.")
    parser.add_argument("--dim", type=int, default=64, help="Fake model hidden size.")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--start-layer", type=int, default=7)
    parser.add_argument("--end-layer", type=int, default=21)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if not args.dry_run and args.config is None:
        parser.error("Provide --config for a real backend validation, or pass --dry-run.")
    return args


def main() :
    args = parse_args()
    if args.dry_run:
        run_dry_run(args)
    else:
        run_config_validation(args)


if __name__ == "__main__":
    main()
