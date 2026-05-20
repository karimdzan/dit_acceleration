"""FID computation via clean-fid."""
import os
from pathlib import Path


def precompute_reference_stats(
    image_dir: str | Path,
    save_path: str | Path,
    mode: str = "clean",
    device: str = "cuda",
) -> None:
    from cleanfid import fid

    image_dir = str(image_dir)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    name = save_path.stem
    fid.make_custom_stats(name, image_dir, mode=mode, device=device, num_workers=4)
    print(f"Reference stats registered under clean-fid name={name!r}")


def _sweep_tmp_files(samples_dir: Path) -> int:
    """Remove stray *.png.tmp files left by an interrupted writer."""
    swept = 0
    for tmp in samples_dir.glob("*.png.tmp"):
        try:
            os.remove(tmp)
            swept += 1
        except OSError:
            pass
    return swept


def _verify_samples_dir(samples_dir: Path, max_check: int | None = None) -> list[str]:
    from PIL import Image

    files = sorted(samples_dir.glob("*.png"))
    if max_check is not None:
        files = files[:max_check]

    bad: list[str] = []
    for p in files:
        try:
            with Image.open(p) as im:
                im.load()
        except Exception as e:
            bad.append(f"{p}: {type(e).__name__}: {e}")
    return bad


def compute_fid(
    samples_dir: str | Path,
    ref_name: str,
    mode: str = "clean",
    device: str = "cuda",
    *,
    pre_validate: bool = False,
) -> float:
    from cleanfid import fid

    samples_dir = Path(samples_dir)

    if pre_validate:
        swept = _sweep_tmp_files(samples_dir)
        if swept:
            print(f"  swept {swept} stray *.png.tmp files from {samples_dir}")
        bad = _verify_samples_dir(samples_dir)
        if bad:
            preview = "\n  ".join(bad[:10])
            extra = f"\n  ... and {len(bad) - 10} more" if len(bad) > 10 else ""
            raise RuntimeError(
                f"Found {len(bad)} corrupt PNG file(s) in {samples_dir}.\n"
                f"First {min(10, len(bad))} bad file(s):\n  {preview}{extra}"
            )

    return float(
        fid.compute_fid(
            str(samples_dir),
            dataset_name=ref_name,
            mode=mode,
            dataset_split="custom",
            device=device,
            num_workers=4,
        )
    )
