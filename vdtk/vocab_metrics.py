import logging
from collections import Counter
from typing import Optional

import click
import numpy as np
import rich
from rich.progress import track

from vdtk.data_utils import load_dataset


def _compute_head_tokens(vocab_counts, ratio: float = 0.9):
    counts = 0
    total_tokens = sum(vocab_counts.values())
    for i, (token, count) in enumerate(vocab_counts.most_common()):
        counts += count
        if counts / total_tokens >= ratio:
            return i
    return len(vocab_counts)


def _compute_ws_uniqueness(sample_vocabs):
    # Number of tokens which are unique within a sample
    return [len([i for i in sv.values() if i == 1]) / len(sv) for sv in sample_vocabs]


def _compute_bs_uniqueness(vocab_counts, sample_vocabs):
    # Number of tokens which are unique to only one sample per sample
    return [len([i for i in sv.keys() if vocab_counts[i] == 1]) / sum(sv.values()) for sv in sample_vocabs]


def _compute_vocab_stats(vocab_counts, sample_vocabs):
    unique_tokens = len(vocab_counts)
    total_tokens = sum(vocab_counts.values())
    head_tokens = _compute_head_tokens(vocab_counts)
    ws_uniqueness = _compute_ws_uniqueness(sample_vocabs)
    bs_uniqueness = _compute_bs_uniqueness(vocab_counts, sample_vocabs)

    # Print the output table
    vocab_stats_table = rich.table.Table(title="Vocab Base Statistics", title_justify="left")
    vocab_stats_table.add_column("Unique Tokens")
    vocab_stats_table.add_column("Total Tokens")
    vocab_stats_table.add_column("90% Head")
    vocab_stats_table.add_column("Mean WS-Uniqueness")
    vocab_stats_table.add_column("Mean BS-Uniqueness")

    vocab_stats_table.add_row(
        str(unique_tokens),
        str(total_tokens),
        str(head_tokens),
        f"{round(np.mean(ws_uniqueness) * 100, 2)}%",
        f"{round(np.mean(bs_uniqueness) * 100, 2)}%",
    )
    return vocab_stats_table


def _compute_pos_stats(noun_counts, verb_counts, sample_noun_counts, sample_verb_counts):
    pos_stats_table = rich.table.Table(title="POS Base Statistics", title_justify="left")

    unique_nouns = len(noun_counts)
    unique_verbs = len(verb_counts)
    total_nouns = sum(noun_counts.values())
    total_verbs = sum(verb_counts.values())
    head_nouns = _compute_head_tokens(noun_counts)
    head_verbs = _compute_head_tokens(verb_counts)
    ws_noun_uniqueness = _compute_ws_uniqueness(sample_noun_counts)
    ws_verb_uniqueness = _compute_ws_uniqueness(sample_verb_counts)
    bs_noun_uniqueness = _compute_bs_uniqueness(noun_counts, sample_noun_counts)
    bs_verb_uniqueness = _compute_bs_uniqueness(verb_counts, sample_verb_counts)

    pos_stats_table.add_column("Unique Nouns")
    pos_stats_table.add_column("Unique Verbs")
    pos_stats_table.add_column("Total Nouns")
    pos_stats_table.add_column("Total Verbs")
    pos_stats_table.add_column("90% Head Nouns")
    pos_stats_table.add_column("90% Head Verbs")
    pos_stats_table.add_column("Mean WS-Uniqueness Nouns")
    pos_stats_table.add_column("Mean WS-Uniqueness Verbs")
    pos_stats_table.add_column("Mean BS-Uniqueness Nouns")
    pos_stats_table.add_column("Mean BS-Uniqueness Verbs")

    pos_stats_table.add_row(
        str(unique_nouns),
        str(unique_verbs),
        str(total_nouns),
        str(total_verbs),
        str(head_nouns),
        str(head_verbs),
        f"{round(np.mean(ws_noun_uniqueness) * 100, 2)}%",
        f"{round(np.mean(ws_verb_uniqueness) * 100, 2)}%",
        f"{round(np.mean(bs_noun_uniqueness) * 100, 2)}%",
        f"{round(np.mean(bs_verb_uniqueness) * 100, 2)}%",
    )
    return pos_stats_table


def _count_nouns_and_verbs(data):
    noun_counts = Counter()
    verb_counts = Counter()
    sample_noun_counts = []
    sample_verb_counts = []
    for sample in track(data, transient=True, description="Counting Nouns and Verbs"):
        sample_noun_counts.append(Counter())
        sample_verb_counts.append(Counter())
        for reference in sample.references_tokenized_pos:
            for token, pos in reference:
                if pos in ("NOUN", "PROPN"):
                    noun_counts[token] += 1
                    sample_noun_counts[-1][token] += 1
                elif pos == "VERB":
                    verb_counts[token] += 1
                    sample_verb_counts[-1][token] += 1
    return noun_counts, verb_counts, sample_noun_counts, sample_verb_counts


def _count_tokens(data):
    vocab_counts = Counter()
    sample_vocabs = []
    for sample in track(data, transient=True, description="Counting Tokens"):
        sample_vocab = Counter()
        for reference in sample.references_tokenized_text:
            vocab_counts.update(reference)
            sample_vocab.update(reference)
        sample_vocabs.append(sample_vocab)
    return vocab_counts, sample_vocabs


@click.command()
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option("--split", default=None, type=str, help="Split to evaluate")
def vocab_stats(dataset_path: str, split: Optional[str] = None) -> None:

    logging.info("Loading dataset...")
    data = load_dataset(dataset_path)
    if split is not None:
        # Filter the data for the correct split
        data = [s for s in data if s.split == split]

    # Compute the vocab
    logging.info("Tokenizing dataset and counting vocab...")
    vocab_counts, sample_vocabs = _count_tokens(data)

    # Compute counters for POS tags
    logging.info("Counting POS tags...")
    noun_counts, verb_counts, sample_noun_counts, sample_verb_counts = _count_nouns_and_verbs(data)

    logging.info("Computing statistics...")
    vocab_stats_table = _compute_vocab_stats(vocab_counts, sample_vocabs)
    pos_stats_table = _compute_pos_stats(noun_counts, verb_counts, sample_noun_counts, sample_verb_counts)

    console = rich.console.Console()
    console.print()
    console.print(vocab_stats_table)
    console.print(pos_stats_table)
