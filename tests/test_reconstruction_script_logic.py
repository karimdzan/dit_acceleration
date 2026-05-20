from PIL import Image
import numpy as np

from fae.backbones.base import FrozenVisionBackbone


class DummyBackbone(FrozenVisionBackbone):
    input_size = 224

    def preprocess(self, images):
        raise NotImplementedError

    def build_reconstruction_targets(self, images, output_size=None):
        return self._raw_to_reconstruction_targets(images, output_size=output_size)


def test_reconstruction_targets_can_use_decoder_output_size():
    image = Image.fromarray(np.zeros((300, 300, 3), dtype=np.uint8))
    backbone = DummyBackbone()
    targets = backbone.build_reconstruction_targets([image], output_size=256)
    assert tuple(targets.shape) == (1, 3, 256, 256)
