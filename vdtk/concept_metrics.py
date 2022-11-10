import logging
import os
from collections import defaultdict
from typing import Optional, List

import click
import numpy as np
import rich
from fuzzysearch import find_near_matches
from fuzzywuzzy import process
from mpire import WorkerPool
from rich.progress import track, Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from vdtk.data_utils import load_dataset, Sample
from vdtk.metrics.bleu.bleu import Bleu
from vdtk.metrics.rouge.rouge import Rouge

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


def _match_concept(concept, reference, fuzzy=False, fuzzy_threshold=90):
    if fuzzy:
        return len(list(_fuzzy_extract(concept, reference, fuzzy_threshold))) > 0
    return concept in reference


def _compute_overlap(
    data: List[Sample], concept_set: str, fuzzy: bool = False, fuzzy_threshold: int = 90, candidates: bool = False
):
    logging.info(f"Computing overlap with {concept_set}")
    concepts = _load_concepts(CONCEPT_SETS[concept_set])
    matched_captions = defaultdict(set)
    matches = []
    for sample in track(data, transient=True, description=f"Computing {concept_set} overlap"):
        sample_matched = False
        for reference in sample.references if not candidates else sample.candidates:
            for concept in concepts:
                if _match_concept(concept, reference, fuzzy, fuzzy_threshold):
                    sample_matched = True
                    matched_captions[concept].add(reference)
        if sample_matched:
            matches.append(sample)
    return matches, matched_captions


def _leave_one_out(data, concept_overlap, concept_set, fuzzy, fuzzy_threshold, candidates):
    _, matched_concepts = concept_overlap
    logging.info(f"Computing leave-one-out for {concept_set}")

    # Construct a function which will be used to initialize the workers
    def __loo_worker_init_fn(worker_state):
        worker_state["scorers"] = {
            "BLEU": Bleu(4),
            "ROUGE": Rouge(),
            # "METEOR": Meteor(),
        }  # Note: CIDEr doesn't really make sense here...
        worker_state["matched_concepts"] = matched_concepts
        worker_state["candidates"] = candidates

    def __loo_worker(worker_state, sample):
        # Build the hypothesis set
        hypotheses = set()
        references = set((sample.references if not worker_state["candidates"] else sample.candidates))
        for concept in worker_state["matched_concepts"].keys():
            for reference in sample.references if not worker_state["candidates"] else sample.candidates:
                if _match_concept(concept, reference, fuzzy, fuzzy_threshold):
                    hypotheses |= set([h for h in worker_state["matched_concepts"][concept] if h not in references])
        # Compute the scores
        scores = {}
        for scorer_name, scorer in worker_state["scorers"].items():
            # logging.debug(
            #     f"Computing {scorer_name} for sample {sample._id} ({len(references)} references) ({len(hypotheses)} hyps)"
            # )
            scores[scorer_name] = [scorer.compute_score({0: list(references)}, {0: [hyp]}) for hyp in hypotheses]
        # logging.info(f"Computed scores for sample {sample._id}")
        return scores

    # Run the leave one out evaluation
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

        logging.info(f"Computing Leave-One-Out Concept Scores for {len(data)} samples.")
        with Progress(*progress_columns, transient=True) as progress:
            for result in progress.track(
                pool.imap(__loo_worker, data, worker_init=__loo_worker_init_fn),
                description="Evaluating...",
                total=len(data),
            ):
                experimental_results.append(result)

    return experimental_results


@click.command()
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option("--split", default=None, type=str, help="Split to evaluate")
@click.option("--fuzzy", default=False, type=bool, help="Use fuzzy matching")
@click.option("--fuzzy-threshold", default=90, type=click.IntRange(0, 100), help="Fuzzy matching threshold")
@click.option("--candidates", default=False, is_flag=True, help="Evaluate candidates instead of references")
def concept_overlap(
    dataset_path: str,
    split: Optional[str] = None,
    fuzzy: bool = False,
    fuzzy_threshold: int = 90,
    candidates: bool = False,
) -> None:

    logging.info("Loading dataset...")
    data = load_dataset(dataset_path)
    if split is not None:
        # Filter the data for the correct split
        data = [s for s in data if s.split == split]

    # Filter data for samples with references
    data = [s for s in data if (s.references if not candidates else s.candidates)]

    overlaps = {
        k: _compute_overlap(data, k, fuzzy=fuzzy, fuzzy_threshold=fuzzy_threshold, candidates=candidates)
        for k in CONCEPT_SETS.keys()
    }

    # Build the table
    concept_table = Table(title=f"Concept Set Matches ({'Exact' if not fuzzy else 'Fuzzy'})", title_justify="left")
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


@click.command()
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option("--split", default=None, type=str, help="Split to evaluate")
@click.option("--fuzzy", default=False, type=bool, help="Use fuzzy matching")
@click.option("--fuzzy-threshold", default=90, type=click.IntRange(0, 100), help="Fuzzy matching threshold")
@click.option("--candidates", default=False, is_flag=True, help="Evaluate candidates instead of references")
def concept_leave_one_out(
    dataset_path: str,
    split: Optional[str] = None,
    fuzzy: bool = False,
    fuzzy_threshold: int = 90,
    candidates: bool = False,
) -> None:

    logging.info("Loading dataset...")
    data = load_dataset(dataset_path)
    if split is not None:
        # Filter the data for the correct split
        data = [s for s in data if s.split == split]
    # Filter data for samples with references
    data = [s for s in data if (s.references if not candidates else s.candidates)]

    overlaps = {
        k: _compute_overlap(data, k, fuzzy=fuzzy, fuzzy_threshold=fuzzy_threshold, candidates=candidates)
        for k in CONCEPT_SETS.keys()
    }
    leave_one_out = {
        k: _leave_one_out(data, overlaps[k], k, fuzzy, fuzzy_threshold, candidates) for k in CONCEPT_SETS.keys()
    }

    # Build the table
    concept_table = Table(
        title=f"Concept Set Leave-One-Out ({'Exact' if not fuzzy else 'Fuzzy'})", title_justify="left"
    )
    concept_table.add_column("Concept Set", justify="left")
    concept_table.add_column("% Matches", justify="right")
    concept_table.add_column("BLEU@1", justify="right")
    concept_table.add_column("BLEU@2", justify="right")
    concept_table.add_column("BLEU@3", justify="right")
    concept_table.add_column("BLEU@4", justify="right")
    concept_table.add_column("ROUGE-L", justify="right")
    # concept_table.add_column("METEOR", justify="right")

    for concept_set, concept_results in leave_one_out.items():
        bleu_1_scores = [[s[0][0] for s in r["BLEU"]] for r in concept_results]
        bleu_2_scores = [[s[0][1] for s in r["BLEU"]] for r in concept_results]
        bleu_3_scores = [[s[0][2] for s in r["BLEU"]] for r in concept_results]
        bleu_4_scores = [[s[0][3] for s in r["BLEU"]] for r in concept_results]
        rouge_scores = [[s[0] for s in r["ROUGE"]] for r in concept_results]
        # meteor_scores = [[s[0] for s in r["METEOR"]] for r in concept_results]

        bb1 = [np.amax(s) for s in bleu_1_scores if len(s) > 0]
        bb2 = [np.amax(s) for s in bleu_2_scores if len(s) > 0]
        bb3 = [np.amax(s) for s in bleu_3_scores if len(s) > 0]
        bb4 = [np.amax(s) for s in bleu_4_scores if len(s) > 0]
        rr = [np.amax(s) for s in rouge_scores if len(s) > 0]
        # mm = [np.amax(s) for s in meteor_scores if len(s) > 0]

        concept_table.add_row(
            concept_set,
            f"{(len(overlaps[concept_set][0]) / len(data)) * 100:.2f}%",
            f"{np.mean(bb1):.2f} +/- {np.std(bb1):.2f}",
            f"{np.mean(bb2):.2f} +/- {np.std(bb2):.2f}",
            f"{np.mean(bb3):.2f} +/- {np.std(bb3):.2f}",
            f"{np.mean(bb4):.2f} +/- {np.std(bb4):.2f}",
            f"{np.mean(rr):.2f} +/- {np.std(rr):.2f}",
            # f"{np.mean(mm):.2f} +/- {np.std(mm):.2f}",
        )

    console = rich.console.Console()
    console.print()
    console.print(concept_table)
