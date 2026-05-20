import torch

from fae.profiling.runtime import RuntimeProfiler, patch_diffusion_pipeline_components, summarize_runs


class DummyModule:
    def __init__(self, value=1):
        self.value = value

    def forward(self, x):
        return x + self.value


class DummyVAE:
    def decode(self, x, return_dict=False):
        return (x * 2,)


class DummyPipe:
    def __init__(self):
        self.text_encoder = DummyModule(1)
        self.transformer = DummyModule(2)
        self.vae = DummyVAE()
        self.scheduler = None
        self.image_processor = None

    def encode_prompt(self, prompt):
        return self.text_encoder.forward(torch.tensor([1.0]))

    def __call__(self, prompt="x"):
        embeds = self.encode_prompt(prompt)
        hidden = self.transformer.forward(embeds)
        image = self.vae.decode(hidden, return_dict=False)[0]
        return image


def test_runtime_profiler_patches_components():
    pipe = DummyPipe()
    profiler = RuntimeProfiler(device="cpu", use_cuda_events=False)
    patched = patch_diffusion_pipeline_components(profiler, pipe)
    assert "encode_prompt_total" in patched
    assert "transformer_forward" in patched
    assert "vae_decode" in patched

    out = pipe("hello")
    assert torch.allclose(out, torch.tensor([8.0]))
    snap = profiler.snapshot()
    assert snap["encode_prompt_total"]["calls"] == 1
    assert snap["text_encoder_forward"]["calls"] == 1
    assert snap["transformer_forward"]["calls"] == 1
    assert snap["vae_decode"]["calls"] == 1

    profiler.restore()


def test_summarize_runs_excludes_warmup():
    runs = [
        {"run_index": 0, "warmup": True, "batch_size": 1, "total_wall_ms": 100.0, "components": {}},
        {
            "run_index": 1,
            "warmup": False,
            "batch_size": 1,
            "total_wall_ms": 10.0,
            "components": {"transformer_forward": {"total_wall_ms": 6.0, "calls": 2}},
        },
        {
            "run_index": 2,
            "warmup": False,
            "batch_size": 1,
            "total_wall_ms": 20.0,
            "components": {"transformer_forward": {"total_wall_ms": 12.0, "calls": 2}},
        },
    ]
    summary = summarize_runs(runs, warmup_runs=1)
    assert summary["num_runs_measured"] == 2
    assert summary["total_wall_ms"]["mean"] == 15.0
    assert summary["components"]["transformer_forward"]["mean"] == 9.0
