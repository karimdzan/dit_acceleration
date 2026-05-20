# RAE + Sana Sprint

## 1. Reproduce official RAE reconstruction quality first

The current official NYU/VISIONx RAE checkpoints are available in two layouts:

- **Recommended reproduction path:** standalone Diffusers repos such as `nyu-visionx/RAE-dinov2-wReg-base-ViTXL-n08`. These expose `AutoencoderRAE` in `config.json` and store weights as `diffusion_pytorch_model.safetensors`.
- **Archival/original bundle:** `nyu-visionx/RAE-collections`, a large bundle with legacy PyTorch files such as `decoders/dinov2/wReg_base/ViTXL_n08/model.pt`, `DiTs/.../stage2_model.pt`, and `stats/`.

### Build a deterministic ImageNet-1K 256 subsample

The data is expected in class folders such as `/tmp/data/abacus`, `/tmp/data/...`.

```bash
python -m fae.scripts.make_imagenet256_subset \
  --data-root /tmp/data \
  --out runs/imagenet256_subset_8pc \
  --samples-per-class 8 \
  --seed 0 \
  --symlink
```

This writes `manifest.csv` and `summary.json`. With `--symlink`, it also creates an ImageNet-like subset under `runs/imagenet256_subset_8pc/images/`.

### Load official RAE from Hugging Face and measure reconstruction FID

```bash
python -m fae.scripts.eval_rae_reconstruction_fid \
  --manifest runs/imagenet256_subset_8pc/manifest.csv \
  --model-id nyu-visionx/RAE-dinov2-wReg-base-ViTXL-n08 \
  --batch-size 8 \
  --dtype bf16 \
  --out runs/rae_rfid_8pc \
  --save-png
```

Outputs:

- `metrics.json` with rFID and latent summary stats
- optional `real_png/` and `recon_png/` folders for visual inspection

Install requirements:

```bash
pip install -U torch torchvision diffusers transformers accelerate safetensors torchmetrics torch-fidelity pillow tqdm
```

## 2. Apply RAE to Sana-Sprint

The first complete integration is a frozen-model latent bridge:

- freeze the official HF `AutoencoderRAE`
- freeze Sana-Sprint VAE and transformer
- encode each image with RAE
- encode the same image with Sana-Sprint VAE
- train `ConvBridge(RAE latent -> Sana latent)` by MSE
- save `rae_to_sana_bridge.pt`

```bash
python -m fae.scripts.train_sana_sprint_rae_bridge \
  --manifest runs/imagenet256_subset_8pc/manifest.csv \
  --rae-model-id nyu-visionx/RAE-dinov2-wReg-base-ViTXL-n08 \
  --sana-model-id Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers \
  --image-size 1024 \
  --batch-size 4 \
  --steps 1000 \
  --out runs/sana_rae_bridge
```
