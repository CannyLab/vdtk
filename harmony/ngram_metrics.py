import itertools
import logging
from typing import Optional

import click
import numpy as np
import rich
from rich.progress import track

from harmony.data_utils import load_dataset
from harmony.lm import NGramLM


def _compute_evs(model: NGramLM) -> float:
    return 1 - len([i for i in model.model.keys() if len(model.model[i]) <= 1]) / len(model.model)


def _compute_ed(ls, evs_2, evs_3, evs_4):
    if ls == 0:
        return 1
    if ls == 1:
        return (1 - evs_2) * _compute_ed(0, evs_2, evs_3, evs_4) + (evs_2) * (_compute_ed(0, evs_2, evs_3, evs_4) + 1)
    if ls == 2:
        return (1 - evs_3) * _compute_ed(1, evs_2, evs_3, evs_4) + (evs_3) * (_compute_ed(1, evs_2, evs_3, evs_4) + 1)
    return (1 - evs_4) * _compute_ed(ls - 1, evs_2, evs_3, evs_4) + (evs_4) * (
        _compute_ed(ls - 1, evs_2, evs_3, evs_4) + 1
    )


@click.command()
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option("--split", default=None, type=str, help="Split to evaluate")
def ngram_stats(dataset_path: str, split: Optional[str] = None) -> None:

    logging.info("Loading dataset...")
    data = load_dataset(dataset_path)
    if split is not None:
        # Filter the data for the correct split
        data = [s for s in data if s.split == split]
    # Filter data for samples with references
    data = [s for s in data if s.references]

    # Build the n-gram language models
    logging.info("Tokenizing samples...")
    tokenized_data = list(
        itertools.chain.from_iterable(
            [
                sample.references_tokenized_text
                for sample in track(data, transient=True, description="Tokenizing Samples")
            ]
        )
    )
    logging.info("Building language models...")
    two_lm = NGramLM(tokenized_data, 2)
    three_lm = NGramLM(tokenized_data, 3)
    four_lm = NGramLM(tokenized_data, 4)

    logging.info("Computing statistics...")

    two_ll = np.mean([two_lm.log_likelihood(t) for t in tokenized_data])
    three_ll = np.mean([three_lm.log_likelihood(t) for t in tokenized_data])
    four_ll = np.mean([four_lm.log_likelihood(t) for t in tokenized_data])

    evs_2 = _compute_evs(two_lm)
    evs_3 = _compute_evs(three_lm)
    evs_4 = _compute_evs(four_lm)

    ll_table = rich.table.Table(title="N-Gram Model Quality", title_justify="left")
    ll_table.add_column("N")
    ll_table.add_column("Log Likelihood")
    ll_table.add_column("Perplexity")
    ll_table.add_row("2", str(two_ll), str(np.exp(-two_ll)))
    ll_table.add_row("3", str(three_ll), str(np.exp(-three_ll)))
    ll_table.add_row("4", str(four_ll), str(np.exp(-four_ll)))

    evs_title = "EVS@N (Essential Vocab Size @ N)"
    evs_table = rich.table.Table(title=evs_title, title_justify="left", min_width=min(80, len(evs_title)))
    evs_table.add_column("N")
    evs_table.add_column("EVS@N")

    evs_table.add_row("2", f"{evs_2 * 100:.2f}%")
    evs_table.add_row("3", f"{evs_3 * 100:.2f}%")
    evs_table.add_row("4", f"{evs_4 * 100:.2f}%")

    ed_title = "ED@N (Expected Number of Decisions @ N)"
    ed_table = rich.table.Table(title=ed_title, title_justify="left", min_width=min(80, len(ed_title)))
    ed_table.add_column("N")
    ed_table.add_column("ED@N")
    for i in range(5, 25, 5):
        ed_table.add_row(str(i), f"{_compute_ed(i, evs_2, evs_3, evs_4):.2f}")

    console = rich.console.Console()
    console.print()
    console.print(ll_table)
    console.print(evs_table)
    console.print(ed_table)
