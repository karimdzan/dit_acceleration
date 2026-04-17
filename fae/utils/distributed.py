import os
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def init_distributed_from_env() -> bool:
    if not dist.is_available() or dist.is_initialized():
        return is_distributed()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    return True


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def maybe_wrap_ddp(model: torch.nn.Module | None, device: torch.device) -> torch.nn.Module | None:
    if model is None or not is_distributed():
        return model
    if not any(p.requires_grad for p in model.parameters()):
        return model
    kwargs: dict[str, Any] = {
        "broadcast_buffers": False,
        "find_unused_parameters": False,
    }
    if device.type == "cuda":
        kwargs["device_ids"] = [device.index]
        kwargs["output_device"] = device.index
    return DDP(model, **kwargs)


def maybe_set_dataloader_epoch(dataloader, epoch: int) -> None:
    sampler = getattr(dataloader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def reduce_mean_dict(values: dict[str, float], count: int, device: torch.device, sum_keys: set[str] | None = None) -> dict[str, float]:
    sum_keys = sum_keys or set()
    if not is_distributed():
        denom = max(count, 1)
        return {k: (float(v) if k in sum_keys else float(v) / denom) for k, v in values.items()}

    keys = list(values.keys())
    tensor = torch.tensor([float(values[k]) for k in keys] + [float(count)], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    total_count = max(int(tensor[-1].item()), 1)
    reduced: dict[str, float] = {}
    for i, key in enumerate(keys):
        value = float(tensor[i].item())
        reduced[key] = value if key in sum_keys else value / total_count
    return reduced
