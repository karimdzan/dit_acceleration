import numpy as np
from PIL import Image

from fae.backbones.base import FrozenVisionBackbone


class DummyBackbone(FrozenVisionBackbone):
    def preprocess(self, images):
        raise NotImplementedError

    def build_reconstruction_targets(self, images, output_size=None):
        return self._raw_to_reconstruction_targets(images, output_size=output_size)

    def forward_features(self, inputs):
        raise NotImplementedError


def test_raw_reconstruction_targets_honor_output_size_and_range():
    backbone = DummyBackbone()
    backbone.input_size = 224
    img = Image.fromarray(np.full((300, 280, 3), 255, dtype=np.uint8))
    out = backbone.build_reconstruction_targets([img], output_size=256)
    assert tuple(out.shape) == (1, 3, 256, 256)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0
