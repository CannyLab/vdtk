import base64
import glob
import io
import json
import os
import sys
import tempfile

import pytest
from click.testing import CliRunner

from vdtk.clip_recall import clip_recall


def test_clip_recall() -> None:
    if len(list(glob.glob(os.path.join(os.path.dirname(__file__), "test_assets", "coco_test_images/*.jpg")))) < 1:
        pytest.skip("COCO Test Images not downloaded...")

    dataset_file = os.path.join(os.path.dirname(__file__), "test_assets", "dataset_small.json")
    output_bytes = run_cmd_to_get_output_bytes(dataset_file)
    print(output_bytes)

    assert "0.9600" in output_bytes  # Recall@1 References
    assert "1.3000" in output_bytes  # MR (candidate)
    assert "0.2291" in output_bytes  # MRR Candidate Stddev
    assert "4.0000" in output_bytes  # 100% recall (references)


def test_clip_recall_b64() -> None:
    dataset_file = os.path.join(os.path.dirname(__file__), "test_assets", "dataset_small.json")
    media_root = os.path.join(os.path.dirname(__file__), "test_assets", "coco_test_images")

    # build b64 dataset
    with open(dataset_file, "r") as f:
        dataset = json.load(f)
    for item in dataset:
        bin_image = open(os.path.join(media_root, item["media_path"]), "rb").read()
        b64_image = base64.b64encode(bin_image).decode(sys.getdefaultencoding())
        item["media_b64"] = b64_image
        item["media_path"] = None

    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_file_b64 = os.path.join(temp_dir, os.path.basename(dataset_file))
        with open(dataset_file_b64, "w") as f:
            json.dump(dataset, f)
        b64_output_bytes = run_cmd_to_get_output_bytes(dataset_file_b64)

        dataset_file = os.path.join(os.path.dirname(__file__), "test_assets", "dataset_small.json")
        original_output_bytes = run_cmd_to_get_output_bytes(dataset_file)

        print(original_output_bytes)
        print(b64_output_bytes)

        assert original_output_bytes == b64_output_bytes


def run_cmd_to_get_output_bytes(dataset_file: str) -> bytes:
    runner = CliRunner()
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
    return result.stdout_bytes.decode("utf-8")
