"""FID/Inception Score helpers for generated image folders."""

from pathlib import Path
from typing import Any


def calculate_fid_is_with_torch_fidelity(
    real_dir: str | Path,
    fake_dir: str | Path,
    batch_size: int = 64,
    cuda: bool = True,
    num_workers: int = 4,
    verbose: bool = True,
    compute_kid: bool = False,
) -> dict[str, Any]:
    """Compute FID and Inception Score with torch-fidelity"""

    try:
        import torch_fidelity
    except Exception as exc:
        raise ImportError(
            "torch-fidelity is required for FID/IS. Install with `pip install torch-fidelity`."
        ) from exc

    real_dir = Path(real_dir)
    fake_dir = Path(fake_dir)
    if not real_dir.exists():
        raise FileNotFoundError(f"Real image directory does not exist: {real_dir}")
    if not fake_dir.exists():
        raise FileNotFoundError(f"Fake image directory does not exist: {fake_dir}")

    metrics = torch_fidelity.calculate_metrics(
        input1=str(fake_dir),
        input2=str(real_dir),
        cuda=bool(cuda),
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        isc=True,
        fid=True,
        kid=bool(compute_kid),
        verbose=bool(verbose),
    )
    return {str(k): float(v) if hasattr(v, "__float__") else v for k, v in metrics.items()}
