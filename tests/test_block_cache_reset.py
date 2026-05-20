"""Regression tests for the block-cache per-batch reset."""
import torch

from dit_accel.caching.block_feature_cache import (
    BlockFeatureCacheStore,
    _make_wrapped_forward,
)


def _fake_block_forward(hidden_states, *args, **kwargs):
    return hidden_states + 0.01


def test_clear_for_generation_preserves_stats():
    store = BlockFeatureCacheStore(schedule={(0, 1): True})
    store.hits = 42
    store.misses = 100
    store.step_idx = 7
    store.residuals[0] = torch.zeros(1, 1, 4)

    store.clear_for_generation()

    assert store.step_idx == 0
    assert len(store.residuals) == 0
    assert store.hits == 42
    assert store.misses == 100


def test_full_clear_resets_everything():
    store = BlockFeatureCacheStore(schedule={(0, 1): True})
    store.hits = 42
    store.misses = 100
    store.step_idx = 7
    store.residuals[0] = torch.zeros(1, 1, 4)

    store.clear()

    assert store.step_idx == 0
    assert store.hits == 0
    assert store.misses == 0
    assert len(store.residuals) == 0


def test_cache_hits_across_multiple_simulated_batches():
    n_steps = 4
    schedule = {(0, s): True for s in range(1, n_steps)}
    store = BlockFeatureCacheStore(schedule=schedule)

    wrapped = _make_wrapped_forward(_fake_block_forward, block_idx=0, store=store)
    h = torch.zeros(1, 4, 8)

    def run_one_batch():
        store.step_idx = 0
        for s in range(n_steps):
            _ = wrapped(h)
            store.advance_step()

    run_one_batch()
    assert store.hits == 3
    assert store.misses == 1

    store.clear_for_generation()
    run_one_batch()
    assert store.hits == 6
    assert store.misses == 2

    store.clear_for_generation()
    run_one_batch()
    assert store.hits == 9
    assert store.misses == 3


def test_unfixed_behaviour_reproduces_bug():
    n_steps = 4
    schedule = {(0, s): True for s in range(1, n_steps)}
    store = BlockFeatureCacheStore(schedule=schedule)
    wrapped = _make_wrapped_forward(_fake_block_forward, block_idx=0, store=store)
    h = torch.zeros(1, 4, 8)

    def run_one_batch_without_reset():
        for s in range(n_steps):
            _ = wrapped(h)
            store.advance_step()

    run_one_batch_without_reset()
    hits_after_batch_1 = store.hits
    run_one_batch_without_reset()
    # Without reset, step_idx moves past the schedule range so no further hits.
    assert store.hits == hits_after_batch_1


def main():
    test_clear_for_generation_preserves_stats()
    test_full_clear_resets_everything()
    test_cache_hits_across_multiple_simulated_batches()
    test_unfixed_behaviour_reproduces_bug()
    print("OK")


if __name__ == "__main__":
    main()
