import glob
import os

import pytest
from click.testing import CliRunner

from vdtk.concept_metrics import concept_leave_one_out, concept_overlap


def test_concept_overlap() -> None:
    runner = CliRunner()
    dataset_file = os.path.join(os.path.dirname(__file__), "test_assets", "dataset_small.json")
    result = runner.invoke(
        concept_overlap,
        [
            dataset_file,
        ],
    )
    assert result.exit_code == 0


def test_concept_overlap_fuzzy() -> None:
    runner = CliRunner()
    dataset_file = os.path.join(os.path.dirname(__file__), "test_assets", "dataset_small.json")
    result = runner.invoke(
        concept_overlap,
        [dataset_file, "--fuzzy"],
    )
    assert result.exit_code == 0


def test_concept_leave_one_out() -> None:
    runner = CliRunner()
    dataset_file = os.path.join(os.path.dirname(__file__), "test_assets", "dataset_small.json")
    result = runner.invoke(
        concept_leave_one_out,
        [
            dataset_file,
        ],
    )
    assert result.exit_code == 0


def test_concept_leave_one_out_fuzzy() -> None:
    runner = CliRunner()
    dataset_file = os.path.join(os.path.dirname(__file__), "test_assets", "dataset_small.json")
    result = runner.invoke(
        concept_leave_one_out,
        [dataset_file, "--fuzzy"],
    )
    assert result.exit_code == 0
