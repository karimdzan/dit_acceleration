"""CLIP score via open_clip (ViT-L/14 OpenAI by default)."""
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm


@torch.no_grad()
def compute_clip_score(
    samples_dir: str | Path,
    prompts: list[str],
    device: str = "cuda",
    model_name: str = "ViT-L-14",
    pretrained: str = "openai",
    batch_size: int = 64,
    return_n_scored: bool = False,
):
    """Mean cosine(image, prompt) * 100 between OpenAI CLIP ViT-L/14 features."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()

    samples_dir = Path(samples_dir)

    surviving: list[tuple[int, str, Path]] = []
    missing = 0
    for idx in range(len(prompts)):
        p = samples_dir / f"{idx:08d}.png"
        if p.exists():
            surviving.append((idx, prompts[idx], p))
        else:
            missing += 1
    if missing:
        print(f"  compute_clip_score: {missing}/{len(prompts)} expected images missing")

    if not surviving:
        raise RuntimeError(f"No matching images in {samples_dir}.")

    scores: list[float] = []
    for start in tqdm(range(0, len(surviving), batch_size)):
        chunk = surviving[start:start + batch_size]
        batch_prompts = [c[1] for c in chunk]
        batch_paths = [c[2] for c in chunk]

        pil_imgs: list = []
        kept_prompts: list[str] = []
        for prm, pth in zip(batch_prompts, batch_paths):
            try:
                pil_imgs.append(preprocess(Image.open(pth).convert("RGB")))
                kept_prompts.append(prm)
            except (OSError, ValueError) as e:
                print(f"  skipping {pth.name}: {type(e).__name__}: {e}")

        if not pil_imgs:
            continue

        images = torch.stack(pil_imgs).to(device)
        text_tokens = tokenizer(kept_prompts).to(device)

        img_feat = model.encode_image(images)
        txt_feat = model.encode_text(text_tokens)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        cos = (img_feat * txt_feat).sum(dim=-1)
        scores.extend((cos * 100.0).cpu().tolist())

    if not scores:
        raise RuntimeError(f"Every image in {samples_dir} failed to decode.")

    score = float(sum(scores) / len(scores))
    if return_n_scored:
        return score, len(scores)
    return score
