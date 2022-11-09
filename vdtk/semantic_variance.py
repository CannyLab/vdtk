import itertools
import logging
from typing import Optional

import click
import numpy as np
from rich.progress import track
from rich.table import Table
from rich.console import Console

from vdtk.data_utils import load_dataset
from vdtk.stats_utils import descr


@click.command()
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option("--split", default=None, type=str, help="Split to evaluate")
@click.option("--candidates", default=False, is_flag=True, help="Evaluate candidates instead of references")
def semantic_variance(dataset_path: str, split: Optional[str] = None, candidates: bool = False) -> None:

    logging.info("Loading dataset...")
    data = load_dataset(dataset_path)
    if split is not None:
        # Filter the data for the correct split
        data = [s for s in data if s.split == split]

    # Filter data for samples with references
    data = [s for s in data if (s.references if not candidates else s.candidates)]

    # Compute the semantic variance
    distances = []
    for sample in track(data, description="Computing reference distances", transient=True):
        sample_distances = {}
        if len((sample.references if not candidates else sample.candidates)) > 2:
            for cp_a, emb_a in zip(
                (sample.references if not candidates else sample.candidates),
                (sample.reference_embeddings if not candidates else sample.candidate_embeddings),
            ):
                sample_distances[cp_a] = {}
                for cp_b, emb_b in zip(
                    (sample.references if not candidates else sample.candidates),
                    (sample.reference_embeddings if not candidates else sample.candidate_embeddings),
                ):
                    if cp_a != cp_b:
                        sample_distances[cp_a][cp_b] = 1 - np.dot(emb_a, emb_b) / (
                            np.linalg.norm(emb_a) * np.linalg.norm(emb_b)
                        )
            distances.append(sample_distances)
        else:
            distances.append({})
    logging.info(f"Computed within-sample reference distances for {len(distances)} samples.")

    # Compute the scores for each pair of references
    min_scores = []
    max_scores = []
    mean_scores = []
    aggregate_scores = []
    variance_scores = []
    for sample in track(distances, description="Aggregating scores"):
        aggregate = []
        for caption, values in sample.items():
            if len(values) > 0:
                min_scores.append(np.amin(list(values.values())))
                max_scores.append(np.amax(list(values.values())))
                mean_scores.append(np.mean(list(values.values())))
                aggregate.append(np.min(list(values.values())))
                variance_scores.append(np.var(list(values.values())))

        aggregate_scores.append(np.mean(aggregate))
    logging.info(f"Aggregated scores for {len(distances)} samples.")

    # Print the results
    table = Table(title="Within-Sample Pairwise Embedding Distances", title_justify="left")
    table.add_column("Aggregate")
    table.add_column("Mean")
    table.add_column("Median")
    table.add_column("Min")
    table.add_column("Max")
    table.add_column("Std. Dev.")
    table.add_column("25% Quantile")
    table.add_column("75% Quantile")
    table.add_column("95% Confidence Interval")

    for aggregate in zip(
        [
            "Minimum Pairwise Distance",
            "Maximum Pairwise Distance",
            "Mean Pairwise Distance",
            "Pairwise Distance Variance",
        ],
        [min_scores, max_scores, mean_scores, variance_scores],
    ):
        _stats = descr(aggregate[1])
        table.add_row(
            aggregate[0],
            f"{_stats['mean']:.2f}",
            f"{_stats['median']:.2f}",
            f"{_stats['min']:.2f}",
            f"{_stats['max']:.2f}",
            f"{_stats['stddev']:.2f}",
            f"{_stats['25q']:.2f}",
            f"{_stats['75q']:.2f}",
            f"{_stats['s95ci'][0]:.2f} - {_stats['s95ci'][1]:.2f}", # type: ignore
        )

    console = Console()
    console.print()
    console.print(table)
