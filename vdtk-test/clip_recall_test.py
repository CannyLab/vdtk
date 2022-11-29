import glob
import os

import pytest
from click.testing import CliRunner

from vdtk.clip_recall import clip_recall


def test_clip_recall() -> None:

    if len(list(glob.glob(os.path.join(os.path.dirname(__file__), "test_assets", "coco_test_images/*.jpg")))) < 1:
        pytest.skip("COCO Test Images not downloaded...")

    runner = CliRunner()
    dataset_file = os.path.join(os.path.dirname(__file__), "test_assets", "dataset_small.json")
    result = runner.invoke(
        clip_recall,
        [
            dataset_file,
            "--media-root",
            os.path.join(os.path.dirname(__file__), "test_assets", "coco_test_images"),
        ],
    )
    assert result.exit_code == 0

    # Do some basic assertion testing to make sure the numbers haven't really changed\
    output_bytes = result.stdout_bytes.decode("utf-8")
    assert "0.9600" in output_bytes  # Recall@1 References
    assert "1.3000" in output_bytes  # MR (candidate)
    assert "0.2291" in output_bytes  # MRR Candidate Stddev
    assert "4.0000" in output_bytes  # 100% recall (references)
