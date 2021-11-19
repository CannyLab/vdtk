import itertools
import logging
import time

import click
import nltk
import numpy as np
import rich
from mpire import WorkerPool
from rich.progress import track

from harmony.data_utils import load_dataset

METRIC_FUNCTIONS = {
    "BLEU": lambda x, y: nltk.translate.bleu_score.sentence_bleu(y, x),
    "METEOR": lambda x, y: nltk.translate.meteor_score.meteor_score(y, x),
}


def _greedy_max_coverage(s):
    target_coverage = set(list(itertools.chain.from_iterable(s)))
    covered = set()
    cover = []
    last_coverage_len = 0
    with rich.progress.Progress(transient=True) as progress:
        cover_progress = progress.add_task("Building cover", total=100)
        for i in range(len(s)):
            max_subset_index, max_subset = max(enumerate(s), key=lambda x: len(x[1] - covered))
            cover.append(max_subset_index)
            covered |= max_subset
            remaining = len(target_coverage - covered)
            progress.update(
                cover_progress, advance=int(100 * (len(covered) - last_coverage_len) / len(target_coverage))
            )
            last_coverage_len = len(covered)
            if remaining == 0:
                break
    return cover


@click.command()
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option("--train-split", default="train", type=str, help="Split to use for training captions")
@click.option("--test-split", default="validate", type=str, help="Split to use for testing captions")
@click.option("--metric", default="BLEU", type=click.Choice(["BLEU", "METEOR"]), help="Metric to evaluate")
@click.option("--intervals", default=10, type=click.IntRange(1, None), help="Number of intervals to evaluate on")
def coreset(dataset_path: str, train_split: str, test_split: str, metric: str = "BLEU", intervals: int = 10) -> None:

    logging.info("Loading dataset...")
    data = load_dataset(dataset_path)
    # Filter data for samples with references
    data = [s for s in data if s.references]
    # Get the training data
    train_data = [s for s in data if s.split == train_split]
    # Get the test data
    test_data = [s for s in data if s.split == test_split]

    # Compute the full set of hypotheses
    logging.info("Computing/tokenizing hypotheses...")
    hypotheses = list(
        itertools.chain.from_iterable(
            [
                t.references_tokenized_text
                for t in track(train_data, transient=True, description="Tokenizing Hypotheses")
            ]
        )
    )
    logging.info("Tokenizing test set...")
    validation_samples = [
        s.references_tokenized_text for s in track(test_data, transient=True, description="Tokenizing Test Set")
    ]

    _metric_func = METRIC_FUNCTIONS[metric]

    def _test_caption(hypothesis):
        scores = []
        for s in validation_samples:
            scores.append(_metric_func(hypothesis, s))
        return scores

    logging.info("Computing metric scores for all hypotheses...")
    with WorkerPool(n_jobs=None) as pool:
        all_scores = pool.map(_test_caption, [(h,) for h in hypotheses], progress_bar=True)

    # Construct the data array
    logging.info("Constructing pairwise score array...")
    data_array = np.array(all_scores)
    logging.info("Constructed pairwise score array of shape %s", str(data_array.shape))

    logging.info("Computing core sets...")
    covers = []
    cover_sizes = []
    for target_value in np.linspace(0, 1, num=intervals):
        logging.info("Computing core set for target %s value %s", metric, f"{target_value:.2f}")
        legal_sets = [set(np.argwhere(x > target_value).reshape(-1).tolist()) for x in data_array]
        sets = [(i, s) for i, s in enumerate(legal_sets) if len(s) > 0]
        coverage = set(list(itertools.chain.from_iterable([i[1] for i in sets])))
        covers.append(_greedy_max_coverage([s[1] for s in sets]))
        cover_sizes.append(len(covers[-1]) + (data_array.shape[-1] - len(coverage)))
        logging.info(
            "Covering the validation set at %s value %s requires %s samples",
            metric,
            f"{target_value:.2f}",
            cover_sizes[-1],
        )

    table = rich.table.Table(title="Core-Set Required Captions", title_justify="left")
    table.add_column(f"{metric} Score")
    table.add_column("Core-Set Size")
    table.add_column("Core-Set Required %")

    for f, s in zip(np.linspace(0, 1, num=intervals), cover_sizes):
        table.add_row(f"{f:.2f}", str(s), f"{s * 100 /len(test_data):.2f}%")

    console = rich.console.Console()
    console.print()
    console.print(table)
