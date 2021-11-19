import itertools
import logging
from typing import Optional

import click
import numpy as np
import rich
from rich.progress import track

from harmony.data_utils import load_dataset


def descr(a: np.ndarray):
    """Generate descriptive statistics for the array a"""
    return {
        "mean": np.nanmean(a),
        "median": np.nanmedian(a),
        "max": np.nanmax(a),
        "min": np.nanmin(a),
        "stddev": np.nanstd(a),
        "25q": np.quantile(a, 0.25),
        "75q": np.quantile(a, 0.75),
        "s95ci": (
            np.nanmean(a) - 1.96 * np.nanstd(a) / np.sqrt(len(a)),
            np.nanmean(a) + 1.96 * np.nanstd(a) / np.sqrt(len(a)),
        ),
    }


@click.command()
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option("--split", default=None, type=str, help="Split to evaluate")
def semantic_variance(dataset_path: str, split: Optional[str] = None) -> None:

    logging.info("Loading dataset...")
    data = load_dataset(dataset_path)
    if split is not None:
        # Filter the data for the correct split
        data = [s for s in data if s.split == split]
    # Filter data for samples with references
    data = [s for s in data if s.references]

    # Compute the semantic variance
    distances = []
    for sample in track(data, description="Computing reference distances", transient=True):
        sample_distances = {}
        if len(sample.references) > 2:
            for cp_a, emb_a in zip(sample.references, sample.reference_embeddings):
                sample_distances[cp_a] = {}
                for cp_b, emb_b in zip(sample.references, sample.reference_embeddings):
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
    table = rich.table.Table(title="Within-Sample Pairwise Embedding Distances", title_justify="left")
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
            f"{_stats['s95ci'][0]:.2f} - {_stats['s95ci'][1]:.2f}",
        )

    console = rich.console.Console()
    console.print()
    console.print(table)
