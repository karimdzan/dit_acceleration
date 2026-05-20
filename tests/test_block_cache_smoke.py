"""Smoke tests for the block feature cache (requires GPU + diffusers)."""
import torch


def test_bf16_baseline_vs_empty_schedule():
    from dit_accel.pipeline import load_pipeline, reset_caches

    prompt = "a photo of a golden retriever"
    seed = 7

    pipe_base = load_pipeline(variant="bf16")
    img_a = pipe_base(
        prompt,
        num_inference_steps=4,
        generator=torch.Generator(pipe_base.device).manual_seed(seed),
    ).images[0]

    pipe_cached = load_pipeline(variant="block_cache", cache_schedule={})
    reset_caches(pipe_cached)
    img_b = pipe_cached(
        prompt,
        num_inference_steps=4,
        generator=torch.Generator(pipe_cached.device).manual_seed(seed),
    ).images[0]

    import numpy as np
    a = np.asarray(img_a).astype(np.int32)
    b = np.asarray(img_b).astype(np.int32)
    diff = np.abs(a - b)
    assert diff.mean() < 1.0


def test_calibration_records_all_blocks():
    from dit_accel.pipeline import load_pipeline, reset_caches

    pipe = load_pipeline(variant="block_cache")
    store = pipe._dit_accel_block_cache
    store.calibration_mode = True
    reset_caches(pipe)

    _ = pipe(
        "a photo of a teapot",
        num_inference_steps=4,
        generator=torch.Generator(pipe.device).manual_seed(0),
    ).images

    n_blocks = len(pipe.transformer.transformer_blocks)
    expected_steps = {1, 2, 3}
    seen_steps = {step for (_, step) in store.deltas}
    assert expected_steps.issubset(seen_steps)
    for step in expected_steps:
        seen_blocks = {b for (b, s) in store.deltas if s == step}
        assert len(seen_blocks) == n_blocks


def test_full_skip_does_not_nan():
    from dit_accel.pipeline import load_pipeline, reset_caches

    pipe = load_pipeline(variant="block_cache")
    store = pipe._dit_accel_block_cache
    n_blocks = len(pipe.transformer.transformer_blocks)
    n_steps = 4
    schedule = {(b, s): True for b in range(n_blocks) for s in range(1, n_steps)}
    store.schedule = schedule
    reset_caches(pipe)

    img = pipe(
        "a photo of a sea anemone",
        num_inference_steps=n_steps,
        generator=torch.Generator(pipe.device).manual_seed(0),
    ).images[0]
    import numpy as np
    arr = np.asarray(img)
    assert np.isfinite(arr).all()


def main():
    test_bf16_baseline_vs_empty_schedule()
    test_calibration_records_all_blocks()
    test_full_skip_does_not_nan()
    print("OK")


if __name__ == "__main__":
    main()
