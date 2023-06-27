import logging
import os
from typing import List, Optional, Tuple

import click
import numpy as np
import rich
import spacy
from rich.progress import track
from rich.table import Table

from vdtk.data_utils import load_dataset
from vdtk.score import _handle_baseline_index
from vdtk.utils.nlp import get_or_download_spacy_model
from vdtk.utils.rich import baseline_column


def compute_object_roverlap(
    nlp: spacy.language.Language, query: str, targets: List[str], POS: Tuple[str, ...] = ("NOUN",)
) -> float:
    """Compute the object overlap between a query and a list of targets.

    Args:
        query (str): The query.
        targets (List[str]): The list of targets.

    Returns:
        float: The object overlap.
    """
    query_doc = nlp(query)
    targets_doc = nlp(" ".join(targets))
    query_objects = set([token.text for token in query_doc if token.pos_ in POS])
    targets_objects = set([token.text for token in targets_doc if token.pos_ in POS])
    # Return the recall
    print(query_objects, targets_objects)
    return len(set(query_objects).intersection(set(targets_objects))) / (len(set(targets_objects)) + 1e-8)


def compute_object_rdistance(
    nlp: spacy.language.Language, query: str, targets: List[str], POS: Tuple[str, ...] = ("NOUN",)
) -> float:
    """Compute the object overlap between a query and a list of targets.

    Args:
        query (str): The query.
        targets (List[str]): The list of targets.

    Returns:
        float: The object overlap.
    """
    query_doc = nlp(query)
    targets_doc = nlp(" ".join(targets))
    query_objects = [token for token in query_doc if token.pos_ in POS]
    targets_objects = [token for token in targets_doc if token.pos_ in POS]

    query_uniq = set()
    targets_uniq = set()

    qos = []
    tos = []
    for token in query_objects:
        if token.text not in query_uniq:
            query_uniq.add(token.text)
            qos.append(token)
    for token in targets_objects:
        if token.text not in targets_uniq:
            targets_uniq.add(token.text)
            tos.append(token)

    metric = []
    for q in tos:
        sims = []
        for t in qos:
            sims.append(q.similarity(t))
        metric.append(max(sims) if sims else 0)

    return sum(metric) / (len(metric) + 1e-8)


@click.command()
@click.argument("dataset_paths", type=str, nargs=-1)
@click.option("--split", default=None, type=str, help="Split to evaluate")
def content_recall(
    dataset_paths: List[str],
    split: Optional[str] = None,
) -> None:

    # Get the baseline
    baseline_index, dataset_paths = _handle_baseline_index(dataset_paths)
    _nlp = get_or_download_spacy_model("en_core_web_lg")

    outputs = []
    for ds in track(dataset_paths, transient=True, description="Computing content recall..."):
        data = load_dataset(ds)
        if split is not None:
            # Filter the data for the correct split
            data = [s for s in data if s.split == split]

        if len(data) == 0:
            logging.error(f"Dataset {ds} has no samples for split {split}.")
            continue

        noun_recall = [
            [
                compute_object_roverlap(
                    _nlp,
                    c,
                    sample.references,
                    (
                        "NOUN",
                        "PROPN",
                    ),
                )
                for c in sample.candidates
            ]
            for sample in data
        ]
        verb_recall = [
            [
                compute_object_roverlap(
                    _nlp,
                    c,
                    sample.references,
                    ("VERB",),
                )
                for c in sample.candidates
            ]
            for sample in data
        ]
        noun_distance = [
            [
                compute_object_rdistance(
                    _nlp,
                    c,
                    sample.references,
                    (
                        "NOUN",
                        "PROPN",
                    ),
                )
                for c in sample.candidates
            ]
            for sample in data
        ]

        verb_distance = [
            [
                compute_object_rdistance(
                    _nlp,
                    c,
                    sample.references,
                    ("VERB",),
                )
                for c in sample.candidates
            ]
            for sample in data
        ]

        noun_recall_a = np.array(noun_recall)
        verb_recall_a = np.array(verb_recall)
        noun_distance_a = np.array(noun_distance)
        verb_distance_a = np.array(verb_distance)

        outputs.append(
            (
                noun_recall_a,
                verb_recall_a,
                noun_distance_a,
                verb_distance_a,
            )
        )

    # Print the results
    table = Table(title="Content Recall")
    table.add_column("Dataset", justify="left", style="cyan", no_wrap=True)
    table.add_column("Noun Recall", justify="right", style="magenta")
    table.add_column("Verb Recall", justify="right", style="magenta")
    table.add_column("Noun Recall (Fuzzy)", justify="right", style="magenta")
    table.add_column("Verb Recall (Fuzzy)", justify="right", style="magenta")
    for i, (ds, (nr, vr, nd, vd)) in enumerate(zip(dataset_paths, outputs)):
        nr_mean = np.mean(nr, axis=-1)
        vr_mean = np.mean(vr, axis=-1)
        nd_mean = np.mean(nd, axis=-1)
        vd_mean = np.mean(vd, axis=-1)

        if baseline_index is None:
            table.add_row(
                os.path.basename(ds),
                f"{np.mean(nr_mean):.4f} ± {np.std(nr_mean):.4f}",
                f"{np.mean(vr_mean):.4f} ± {np.std(vr_mean):.4f}",
                f"{np.mean(nd_mean):.4f} ± {np.std(nd_mean):.4f}",
                f"{np.mean(vd_mean):.4f} ± {np.std(vd_mean):.4f}",
            )
        else:
            if i == baseline_index:
                table.add_row(
                    os.path.basename(ds),
                    f"{np.mean(nr_mean):.4f} ± {np.std(nr_mean):.4f}",
                    f"{np.mean(vr_mean):.4f} ± {np.std(vr_mean):.4f}",
                    f"{np.mean(nd_mean):.4f} ± {np.std(nd_mean):.4f}",
                    f"{np.mean(vd_mean):.4f} ± {np.std(vd_mean):.4f}",
                )
            else:
                table.add_row(
                    os.path.basename(ds),
                    baseline_column(nr_mean, np.mean(outputs[baseline_index][0], axis=-1)),
                    baseline_column(vr_mean, np.mean(outputs[baseline_index][1], axis=-1)),
                    baseline_column(nd_mean, np.mean(outputs[baseline_index][2], axis=-1)),
                    baseline_column(vd_mean, np.mean(outputs[baseline_index][3], axis=-1)),
                )
    rich.print(table)
