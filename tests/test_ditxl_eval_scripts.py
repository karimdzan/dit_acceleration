from pathlib import Path
import json
import tempfile

from PIL import Image

from fae.evaluation.ditxl_eval_utils import build_side_by_side_outputs, image_path_for_index


def test_build_side_by_side_outputs():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bdir = tmp / "baseline"
        tdir = tmp / "token_merge"
        bdir.mkdir()
        tdir.mkdir()
        labels = [1, 2, 3, 4]
        for i, label in enumerate(labels):
            Image.new("RGB", (32, 32), color=(255, 0, 0)).save(image_path_for_index(bdir, i, label))
            Image.new("RGB", (32, 32), color=(0, 255, 0)).save(image_path_for_index(tdir, i, label))
        out = build_side_by_side_outputs(
            baseline_dir=bdir,
            token_merge_dir=tdir,
            output_dir=tmp / "side_by_side",
            labels=labels,
            n=4,
            id_to_label={1: "a", 2: "b", 3: "c", 4: "d"},
            grid_columns=2,
        )
        assert Path(out["grid_path"]).exists()
        assert len(out["pair_paths"]) == 4
