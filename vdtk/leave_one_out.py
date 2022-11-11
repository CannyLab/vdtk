import logging
import random
from typing import Optional

import click
import numpy as np
from mpire import WorkerPool
from rich.console import Console
from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn, track)
from rich.table import Table

from vdtk.data_utils import load_dataset
from vdtk.metrics.bleu.bleu import Bleu
from vdtk.metrics.cider.cider import Cider
from vdtk.metrics.meteor.meteor import Meteor
from vdtk.metrics.rouge.rouge import Rouge
from vdtk.stats_utils import descr


def _loo_worker_init_fn(worker_state):
    worker_state["scorers"] = {
        "BLEU": Bleu(4),
        "ROUGE": Rouge(),
        "CIDEr": Cider(),
        "METEOR": Meteor(),
    }


def _loo_worker_fn(worker_state, hypotheses, ground_truths):
    return {
        k: worker_state["scorers"][k].compute_score(ground_truths, hypotheses)
        for k in [
            "BLEU",
            "ROUGE",
            "CIDEr",
            "METEOR",
        ]
    }


@click.command()
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option("--split", default=None, type=str, help="Split to evaluate")
@click.option("--iterations", default=750, type=click.IntRange(min=1), help="Number of iterations to run")
@click.option("--max-gt-size", default=None, type=int, help="Maximum number of ground truth sentences to use")
def leave_one_out(
    dataset_path: str,
    split: Optional[str] = None,
    iterations: int = 750,
    max_gt_size: Optional[int] = None,
) -> None:

    logging.info("Loading dataset...")
    data = load_dataset(dataset_path)
    if split is not None:
        # Filter the data for the correct split
        data = [s for s in data if s.split == split]
    # Filter data for samples with references
    data = [s for s in data if s.references]

    # Generate the hypothesis datasets
    experiments = []
    for i in track(range(iterations), description="Generating hypothesis datasets...", transient=True):
        s_idx = (random.randint(0, len(c.references) - 1) for c in data)
        hypotheses = [tuple(c.references_tokenized_text[i]) for c, i in zip(data, s_idx)]
        ground_truths = [
            list(set(tuple(r)) for r in c.references_tokenized_text if tuple(r) != h) for c, h in zip(data, hypotheses)
        ]
        if max_gt_size is not None:
            ground_truths = [random.sample(gt_set, min(len(gt_set), max_gt_size)) for gt_set in ground_truths]
        experiments.append(
            {
                "ground_truths": {i: [" ".join(g) for g in gt] for i, gt in enumerate(ground_truths)},
                "hypotheses": {i: [" ".join(h)] for i, h in enumerate(hypotheses)},
            }
        )
    logging.info("Generated %d hypothesis datasets", len(experiments))

    experimental_results = []
    with WorkerPool(n_jobs=None, use_worker_state=True) as pool:

        # Setup the progress bar
        progress_columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ]

        with Progress(*progress_columns, transient=True) as progress:
            for result in progress.track(
                pool.imap_unordered(_loo_worker_fn, experiments, worker_init=_loo_worker_init_fn),
                description="Evaluating...",
                total=len(experiments),
            ):
                experimental_results.append(result)

    # Compute the average results across the iterations
    average_results = {
        "BLEU@1": descr([r["BLEU"][0][0] for r in experimental_results]),
        "BLEU@2": descr([r["BLEU"][0][1] for r in experimental_results]),
        "BLEU@3": descr([r["BLEU"][0][2] for r in experimental_results]),
        "BLEU@4": descr([r["BLEU"][0][3] for r in experimental_results]),
        "ROUGE": descr([r["ROUGE"][0] for r in experimental_results]),
        "CIDEr": descr([r["CIDEr"][0] for r in experimental_results]),
        "METEOR": descr([r["METEOR"][0] for r in experimental_results]),
    }

    metrics_table = Table(title=f"Leave One Out Metric Scores ({iterations} Iterations)", title_justify="left")
    metrics_table.add_column("Metric", justify="left")
    metrics_table.add_column("Mean")
    metrics_table.add_column("Median")
    metrics_table.add_column("Min")
    metrics_table.add_column("Max")
    metrics_table.add_column("Std. Dev.")
    metrics_table.add_column("25% Quantile")
    metrics_table.add_column("75% Quantile")
    metrics_table.add_column("95% Confidence Interval")

    for key, value in average_results.items():
        metrics_table.add_row(
            key,
            *[
                f"{value['mean']:.2f}",
                f"{value['median']:.2f}",
                f"{value['min']:.2f}",
                f"{value['max']:.2f}",
                f"{value['stddev']:.2f}",
                f"{value['25q']:.2f}",
                f"{value['75q']:.2f}",
                f"{value['s95ci'][0]:.2f} - {value['s95ci'][1]:.2f}",  # type: ignore
            ],
        )

    # Print the results
    console = Console()
    console.print()
    console.print(metrics_table)
