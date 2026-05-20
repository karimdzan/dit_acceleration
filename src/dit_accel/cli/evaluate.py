"""Compute FID and CLIP for a generated samples directory."""
import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from dit_accel.data import load_coco_30k_captions
from dit_accel.evaluation.clip_score import compute_clip_score
from dit_accel.evaluation.fid import compute_fid
from dit_accel.evaluation.imagenet_prompts import build_prompts

from ._common import CONFIG_PATH


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="evaluate")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    samples = Path(cfg.samples_dir)
    output = Path(cfg.output)

    print(f"Computing FID for {samples} against {cfg.ref_name}...")
    fid_val = compute_fid(samples, ref_name=cfg.ref_name)
    print(f"  FID: {fid_val:.4f}")

    clip_val = None
    clip_n_scored = None
    if not cfg.skip_clip:
        print("Computing CLIP score...")
        if cfg.dataset.name == "coco30k":
            prompts = load_coco_30k_captions(cfg.dataset.coco_local_dir)
        else:
            prompts = build_prompts(samples_per_class=cfg.samples_per_class)
        clip_val, clip_n_scored = compute_clip_score(
            samples, prompts,
            model_name=cfg.clip_model_name,
            pretrained=cfg.clip_pretrained,
            return_n_scored=True,
        )
        print(f"  CLIP: {clip_val:.4f}  ({clip_n_scored}/{len(prompts)} scored)")

    manifest_path = samples / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    result = {
        "samples_dir": str(samples),
        "fid": fid_val,
        "clip": clip_val,
        "clip_n_scored": clip_n_scored,
        "manifest": manifest,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
