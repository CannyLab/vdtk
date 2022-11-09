import itertools
import logging
from typing import Optional

import click
import numpy as np
import rich
from rich.progress import track

from vdtk.data_utils import load_dataset
from vdtk.stats_utils import descr


@click.command()
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option("--split", default=None, type=str, help="Split to evaluate")
@click.option("--reference-key", default="references", type=str, help="Reference key to evaluate")
def caption_stats(dataset_path: str, split: Optional[str] = None, reference_key: str = "references") -> None:

    logging.info("Loading dataset...")
    data = load_dataset(dataset_path, reference_key=reference_key)
    if split is not None:
        # Filter the data for the correct split
        data = [s for s in data if s.split == split]

    # Count the tokens per caption
    tokens_per_caption = descr(
        np.array(
            list(
                itertools.chain.from_iterable(
                    [
                        [len(c) for c in s.references_tokenized_lemma]
                        for s in track(data, description="Tokenizing dataset...", transient=True)
                    ]
                )
            )
        )
    )
    unique_captions = [
        set(tuple(c) for c in s.references_tokenized_lemma)
        for s in track(data, description="Tokenizing dataset...", transient=True)
    ]
    captions_per_video = descr([len(s.references) for s in data if s.references])
    unique_captions_per_video = descr([len(u) for u in unique_captions if len(u) > 0])

    # Display the table
    table = rich.table.Table(title="Caption Metrics", title_justify="left")
    table.add_column("Metric")
    table.add_column("Mean")
    table.add_column("Median")
    table.add_column("Min")
    table.add_column("Max")
    table.add_column("Std. Dev.")
    table.add_column("25% Quantile")
    table.add_column("75% Quantile")
    table.add_column("95% Confidence Interval")

    # Tokens per caption
    table.add_row(
        "Tokens per description",
        f"{tokens_per_caption['mean']:.2f}",
        f"{tokens_per_caption['median']:.2f}",
        f"{tokens_per_caption['min']:.2f}",
        f"{tokens_per_caption['max']:.2f}",
        f"{tokens_per_caption['stddev']:.2f}",
        f"{tokens_per_caption['25q']:.2f}",
        f"{tokens_per_caption['75q']:.2f}",
        f"{tokens_per_caption['s95ci'][0]:.2f} - {tokens_per_caption['s95ci'][1]:.2f}",
    )
    table.add_row(
        "Descriptions per video",
        f"{captions_per_video['mean']:.2f}",
        f"{captions_per_video['median']:.2f}",
        f"{captions_per_video['min']:.2f}",
        f"{captions_per_video['max']:.2f}",
        f"{captions_per_video['stddev']:.2f}",
        f"{captions_per_video['25q']:.2f}",
        f"{captions_per_video['75q']:.2f}",
        f"{captions_per_video['s95ci'][0]:.2f} - {captions_per_video['s95ci'][1]:.2f}",
    )
    table.add_row(
        "Unique descriptions per video",
        f"{unique_captions_per_video['mean']:.2f}",
        f"{unique_captions_per_video['median']:.2f}",
        f"{unique_captions_per_video['min']:.2f}",
        f"{unique_captions_per_video['max']:.2f}",
        f"{unique_captions_per_video['stddev']:.2f}",
        f"{unique_captions_per_video['25q']:.2f}",
        f"{unique_captions_per_video['75q']:.2f}",
        f"{unique_captions_per_video['s95ci'][0]:.2f} - {unique_captions_per_video['s95ci'][1]:.2f}",
    )

    console = rich.console.Console()
    console.print()
    console.print(table)
