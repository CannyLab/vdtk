import os

from click.testing import CliRunner

from vdtk.caption_metrics import caption_stats


def test_basic_caption_stats() -> None:
    runner = CliRunner()
    dataset_file = os.path.join(os.path.dirname(__file__), "test_assets", "dataset_small.json")
    result = runner.invoke(
        caption_stats,
        [
            dataset_file,
        ],
    )
    assert result.exit_code == 0


def test_basic_caption_stats_candidates() -> None:
    runner = CliRunner()
    dataset_file = os.path.join(os.path.dirname(__file__), "test_assets", "dataset_small.json")
    result = runner.invoke(
        caption_stats,
        [
            dataset_file,
            "--candidates",
        ],
    )
    assert result.exit_code == 0
