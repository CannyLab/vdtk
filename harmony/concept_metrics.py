import logging
import os
from collections import defaultdict
from typing import Optional

import click
import numpy as np
import rich
import tqdm
from fuzzysearch import find_near_matches
from fuzzywuzzy import process

from harmony.data_utils import load_dataset

CONCEPT_SETS = {
    "Places365": os.path.join(os.path.dirname(__file__), "assets/places_labels.txt"),
    "MS-COCO": os.path.join(os.path.dirname(__file__), "assets/coco_labels.txt"),
    "ImageNet-1K": os.path.join(os.path.dirname(__file__), "assets/imagenet_labels.txt"),
    "Kinetics-400": os.path.join(os.path.dirname(__file__), "assets/kinetics_400_labels.txt"),
    "Kinetics-600": os.path.join(os.path.dirname(__file__), "assets/kinetics_600_labels.txt"),
}


def _load_concepts(concept_set):
    with open(concept_set) as f:
        concepts = [line.strip() for line in f]
    return concepts


def _fuzzy_extract(qs, ls, threshold):
    """fuzzy matches 'qs' in 'ls' and returns list of
    tuples of (word,index)
    """
    for word, _ in process.extractBests(qs, (ls,), score_cutoff=threshold):
        for match in find_near_matches(qs, word, max_l_dist=1):
            match = word[match.start : match.end]
            index = ls.find(match)
            yield (match, index)


def _compute_overlap(data, concept_set, fuzzy: bool = False, fuzzy_threshold: int = 90):
    logging.info(f"Computing overlap with {concept_set}")
    concepts = _load_concepts(CONCEPT_SETS[concept_set])
    matched_captions = defaultdict(set)
    matches = []
    for sample in tqdm.tqdm(data, leave=False):
        sample_matched = False
        for reference in sample.references:
            for concept in concepts:
                if fuzzy:
                    ss = list(_fuzzy_extract(concept, reference, fuzzy_threshold))
                    if len(ss) > 0:
                        sample_matched = True
                        matched_captions[concept].add(reference)
                else:
                    if concept in reference:
                        sample_matched = True
                        matched_captions[concept].add(reference)
        if sample_matched:
            matches.append(sample)
    return matches, matched_captions


@click.command()
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option("--split", default=None, type=str, help="Split to evaluate")
@click.option("--fuzzy", default=False, type=bool, help="Use fuzzy matching")
@click.option("--fuzzy-threshold", default=90, type=click.IntRange(0, 100), help="Fuzzy matching threshold")
def concept_overlap(
    dataset_path: str, split: Optional[str] = None, fuzzy: bool = False, fuzzy_threshold: int = 90
) -> None:

    logging.info("Loading dataset...")
    data = load_dataset(dataset_path)
    if split is not None:
        # Filter the data for the correct split
        data = [s for s in data if s.split == split]
    # Filter data for samples with references
    data = [s for s in data if s.references]

    overlaps = {k: _compute_overlap(data, k, fuzzy=fuzzy, fuzzy_threshold=fuzzy_threshold) for k in CONCEPT_SETS.keys()}

    # Build the table
    concept_table = rich.table.Table(
        title=f"Concept Set Matches ({'Exact' if not fuzzy else 'Fuzzy'})", title_justify="left"
    )
    concept_table.add_column("Concept Set", justify="left")
    concept_table.add_column("# Samples", justify="right")
    concept_table.add_column("# Matches", justify="right")
    concept_table.add_column("% Matches", justify="right")
    concept_table.add_column("Mean Matches / Concept", justify="right")
    concept_table.add_column("Min Matches / Concept", justify="right")
    concept_table.add_column("Max Matches / Concept", justify="right")

    for concept_set, (matches, matched_captions) in overlaps.items():
        concept_table.add_row(
            concept_set,
            f"{len(data)}",
            f"{len(matches)}",
            f"{(len(matches) / len(data)) * 100:.2f}%",
            f"{np.mean([len(v) for v in matched_captions.values()]):.2f}",
            f"{np.min([len(v) for v in matched_captions.values()]):.2f}",
            f"{np.max([len(v) for v in matched_captions.values()]):.2f}",
        )

    console = rich.console.Console()
    console.print()
    console.print(concept_table)
