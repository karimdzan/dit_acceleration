"""Cross-attention KV cache must produce the same outputs as the baseline."""
import torch

from dit_accel.pipeline import load_pipeline, reset_caches


def main():
    prompt = "a photo of a golden retriever"
    seed = 7

    pipe_base = load_pipeline(variant="bf16")
    pipe_cached = load_pipeline(variant="xattn")

    gen_a = torch.Generator("cuda").manual_seed(seed)
    img_a = pipe_base(prompt, num_inference_steps=4, generator=gen_a).images[0]

    reset_caches(pipe_cached)
    gen_b = torch.Generator("cuda").manual_seed(seed)
    img_b = pipe_cached(prompt, num_inference_steps=4, generator=gen_b).images[0]

    import numpy as np
    a = np.asarray(img_a).astype(np.int32)
    b = np.asarray(img_b).astype(np.int32)
    diff = np.abs(a - b)
    print(f"max pixel diff: {diff.max()}  mean diff: {diff.mean():.4f}")
    assert diff.mean() < 1.0


if __name__ == "__main__":
    main()
