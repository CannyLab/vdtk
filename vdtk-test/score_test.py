import io
import os
from contextlib import redirect_stdout

import numpy as np
import pytest

from vdtk.score import (
    _bleu,
    bert_score,
    bleurt,
    ciderd,
    mauve_score,
    meteor,
    mmd_bert,
    mmd_clip,
    mmd_fasttext,
    mmd_glove,
    rouge,
    trm_bert,
    trm_bert_score,
    trm_bleu,
    trm_cider,
    trm_meteor,
    trm_rouge,
)

_STR_METRIC_MAP = {
    "mmd_bert": mmd_bert,
    "mmd_clip": mmd_clip,
    "mmd_fasttext": mmd_fasttext,
    "mmd_glove": mmd_glove,
    "trm_bert": trm_bert,
    "trm_bert_score": trm_bert_score,
    "trm_bleu": trm_bleu,
    "trm_cider": trm_cider,
    "trm_meteor": trm_meteor,
    "trm_rouge": trm_rouge,
    "bert_score": bert_score,
    "bleurt": bleurt,
    "ciderd": ciderd,
    "mauve_score": mauve_score,
    "meteor": meteor,
    "rouge": rouge,
}


@pytest.mark.parametrize(
    "metric",
    [
        "mmd_bert",
        "mmd_clip",
        "mmd_fasttext",
        "mmd_glove",
    ],
)
def test_mmd_score_fn(metric: str) -> None:
    # Get the dataset path
    dataset_file = os.path.join(os.path.dirname(__file__), "test_assets", "dataset_small.json")
    # Run the metric capturing the output
    f = io.StringIO()
    with redirect_stdout(f):
        _STR_METRIC_MAP[metric]([dataset_file], None, True, None)  # type: ignore
    # Check the output

    # Make sure the metrics are not all 0
    output_str = f.getvalue().lower()

    # Make sure that the metrics look like they are being calculated
    assert "0.0000" not in output_str
    assert "0.000" not in output_str
    assert "nan" not in output_str
    assert "inf" not in output_str


@pytest.mark.parametrize(
    "metric",
    [
        "trm_bert",
        "trm_bert_score",
        "trm_bleu",
        "trm_cider",
        "trm_meteor",
        "trm_rouge",
    ],
)
def test_trm_score_fn(metric: str) -> None:
    # Get the dataset path
    dataset_file = os.path.join(os.path.dirname(__file__), "test_assets", "dataset_small.json")
    # Run the metric capturing the output
    f = io.StringIO()
    with redirect_stdout(f):
        # We use a pretty small UK-Sample value for testing
        _STR_METRIC_MAP[metric]([dataset_file], None, True, 5)  # type: ignore
    # Check the output

    # Make sure the metrics are not all 0
    output_str = f.getvalue().lower()

    # Make sure that the metrics look like they are being calculated
    assert "0.0000" not in output_str
    assert "0.000" not in output_str
    assert "nan" not in output_str
    assert "inf" not in output_str


@pytest.mark.parametrize(
    "metric",
    [
        "ciderd",
        "meteor",
        "rouge",
        "bleurt",
        "bert_score",
        "mauve_score",
    ],
)
def test_simple_score_fn(metric: str) -> None:
    # Get the dataset path
    dataset_file = os.path.join(os.path.dirname(__file__), "test_assets", "dataset_small.json")
    # Run the metric capturing the output
    f = io.StringIO()
    with redirect_stdout(f):
        _STR_METRIC_MAP[metric]([dataset_file], None)  # type: ignore
    # Check the output

    # Make sure the metrics are not all 0
    output_str = f.getvalue().lower()

    # Make sure that the metrics look like they are being calculated
    assert "0.0000" not in output_str
    assert "0.000" not in output_str or metric == "mauve_score"  # Mauve has no variance (it's a constant)
    assert "nan" not in output_str
    assert "inf" not in output_str


def test_bleu_score_fn() -> None:
    # Get the dataset path
    dataset_file = os.path.join(os.path.dirname(__file__), "test_assets", "dataset_small.json")
    # Run the metric capturing the output
    f = io.StringIO()
    with redirect_stdout(f):
        avg_scores, _ = _bleu([dataset_file], None)[0]

    # Check the output
    for score in avg_scores:
        assert score > 0.0
        assert not np.isnan(score)
        assert not np.isinf(score)
