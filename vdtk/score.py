import click
import os
import numpy as np
import rich
from rich.table import Table
from typing import Sequence, List, Tuple, Any
import json
from typing import Optional
from pycocoevalcap.cider.cider import Cider

# from vdtk.metrics.bleu.bleu import Bleu
from vdtk.metrics.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from vdtk.metrics.spice.spice import Spice
from vdtk.metrics.tokenizer.ptbtokenizer import PTBTokenizer

PUNCTUATIONS = [
    "''",
    "'",
    "``",
    "`",
    "-LRB-",
    "-RRB-",
    "-LCB-",
    "-RCB-",
    ".",
    "?",
    "!",
    ",",
    ":",
    "-",
    "--",
    "...",
    ";",
]


@click.group()
def score():
    pass


def _pycoco_eval_cap_metric(
    measure: Any, dataset_paths: Sequence[str], split: Optional[str] = None
) -> List[Tuple[float, List[float]]]:
    tokenizer = PTBTokenizer()

    # Load the paths
    scores = []
    for path in dataset_paths:
        # Load the candidates and references from the dataset
        with open(path, "r") as f:
            data = json.load(f)
            candidates = {
                sample.get("_id", i): sample["candidates"]
                for i, sample in enumerate(data)
                if sample.get("split", None) == split or split is None
            }
            references = {
                sample.get("_id", i): sample["references"]
                for i, sample in enumerate(data)
                if sample.get("split", None) == split or split is None
            }
            references = tokenizer.tokenize(references)
            candidates = tokenizer.tokenize(candidates)

        total_score, score = measure().compute_score(references, candidates)
        scores.append((total_score, score))

    return scores


def _pycoco_eval_cap_multi_metric(
    measure: Any, dataset_paths: Sequence[str], split: Optional[str] = None
) -> List[List[Tuple[float, List[float]]]]:
    tokenizer = PTBTokenizer()

    # Load the paths
    scores = [[] for _ in measure]
    for path in dataset_paths:
        # Load the candidates and references from the dataset
        with open(path, "r") as f:
            data = json.load(f)
            candidates = {
                sample.get("_id", i): sample["candidates"]
                for i, sample in enumerate(data)
                if sample.get("split", None) == split or split is None
            }
            references = {
                sample.get("_id", i): sample["references"]
                for i, sample in enumerate(data)
                if sample.get("split", None) == split or split is None
            }
            references = tokenizer.tokenize(references)
            candidates = tokenizer.tokenize(candidates)

        for i, m in enumerate(measure):
            total_score, score = m().compute_score(references, candidates)
            scores[i].append((total_score, score))

    return scores


def _ciderd(dataset_paths: Sequence[str], split: Optional[str] = None) -> List[Tuple[float, List[float]]]:
    return _pycoco_eval_cap_metric(Cider, dataset_paths, split)


def _bleu(
    dataset_paths: Sequence[str], split: Optional[str] = None
) -> List[Tuple[Tuple[float, ...], Tuple[List[float], ...]]]:
    # BLEU is a special case because it generates 4 scores
    tokenizer = PTBTokenizer()

    # Load the paths
    scores = []
    for path in dataset_paths:
        # Load the candidates and references from the dataset
        with open(path, "r") as f:
            data = json.load(f)
            candidates = {
                sample.get("_id", i): sample["candidates"][:1]
                for i, sample in enumerate(data)
                if sample.get("split", None) == split or split is None
            }
            references = {
                sample.get("_id", i): sample["references"]
                for i, sample in enumerate(data)
                if sample.get("split", None) == split or split is None
            }
            references = tokenizer.tokenize(references)
            candidates = tokenizer.tokenize(candidates)

        avg_scores, score = Bleu().compute_score(references, candidates, return_scores=True)
        # avg_scores, score = Bleu(4).compute_score(references, candidates)

        scores.append(
            (
                (
                    avg_scores[0],
                    avg_scores[1],
                    avg_scores[2],
                    avg_scores[3],
                ),
                (
                    score[0],
                    score[1],
                    score[2],
                    score[3],
                ),
            )
        )

    return scores


def _rouge(dataset_paths: Sequence[str], split: Optional[str] = None) -> List[Tuple[float, List[float]]]:
    return _pycoco_eval_cap_metric(Rouge, dataset_paths, split)


def _meteor(dataset_paths: Sequence[str], split: Optional[str] = None) -> List[Tuple[float, List[float]]]:
    return _pycoco_eval_cap_metric(Meteor, dataset_paths, split)


def _spice(dataset_paths: Sequence[str], split: Optional[str] = None) -> List[Tuple[float, List[float]]]:
    return _pycoco_eval_cap_metric(Spice, dataset_paths, split)


def _handle_baseline_index(dataset_paths: Sequence[str]) -> Tuple[Optional[int], List[str]]:
    baseline_path = None
    output_paths = []
    for path in dataset_paths:
        if path.strip().endswith(":baseline"):
            # output_paths.append(path.strip()[:-9])
            baseline_path = path.strip()[:-9]
        else:
            output_paths.append(path)

    if baseline_path is not None:
        return 0, [baseline_path] + output_paths
    return None, output_paths


def _print_table(
    label: str,
    scores: List[Tuple[float, List[float]]],
    dataset_paths: List[str],
    baseline_index: Optional[int],
    spice: bool = False,
):

    if spice:
        # There are some issues with the Spice score
        spice_raw = [[k["All"]["f"] for k in s[1]] for s in scores]  # type: ignore
        scores = [(s[0], k) for s, k in zip(scores, spice_raw)]

    table = Table(title=f"{label} Scores")
    table.add_column("Dataset")
    table.add_column(f"{label}")
    table.add_column(f"Max {label}")
    table.add_column(f"Min {label}")

    # Direct print
    if baseline_index is None:
        for path, score in zip(dataset_paths, scores):
            table.add_row(
                os.path.basename(path),
                f"{score[0]:0.4f} +/- {np.std(score[1]):0.3f}",
                f"{np.max(score[1]):0.4f}",
                f"{np.min(score[1]):0.4f}",
            )
    else:
        # Print with a baseline
        for i, (path, score) in enumerate(zip(dataset_paths, scores)):
            if i == baseline_index:
                table.add_row(
                    os.path.basename(path),
                    # Add relative improvement
                    f"{score[0]:0.4f} +/- {np.std(score[1]):0.3f}",
                    f"{np.max(score[1]):0.4f}",
                    f"{np.min(score[1]):0.4f}",
                    style="bold",
                )
            else:
                # Handle colors for the baseline
                r1_color = "green" if score[0] > scores[baseline_index][0] else "red"
                r2_color = "green" if np.max(score[1]) > np.max(scores[baseline_index][1]) else "red"
                r3_color = "green" if np.min(score[1]) > np.min(scores[baseline_index][1]) else "red"
                table.add_row(
                    os.path.basename(path),
                    # Relative score to the baseline
                    f"[{r1_color}]{score[0]:0.4f} +/- {np.std(score[1]):0.3f} ({'+' if r1_color == 'green' else '-'}{np.abs(score[0] - scores[baseline_index][0]) / (np.amax([score[0], scores[baseline_index][0]])+1e-12) * 100:0.3f}%)[/{r1_color}]",
                    f"[{r2_color}]{np.max(score[1]):0.4f} ({'+' if r2_color == 'green' else '-'}{np.abs(np.max(score[1]) - np.max(scores[baseline_index][1])) / (np.amax([np.max(score[1]), np.max(scores[baseline_index][1])])+1e-12)*100:0.3f}%)[/{r2_color}]",
                    f"[{r3_color}]{np.min(score[1]):0.4f} ({'+' if r3_color == 'green' else '-'}{np.abs(np.min(score[1]) - np.min(scores[baseline_index][1])) / (np.amax([np.min(score[1]), np.min(scores[baseline_index][1])])+1e-12)*100:0.3f}%)[/{r3_color}]",
                )
    rich.print(table)


@click.command()
@click.argument("dataset_paths", type=str, nargs=-1)
@click.option("--split", default=None, type=str, help="Split to evaluate")
def ciderd(dataset_paths, split):
    baseline_index, dataset_paths = _handle_baseline_index(dataset_paths)
    scores = _ciderd(dataset_paths, split)
    _print_table("CIDEr-D", scores, dataset_paths, baseline_index)


@click.command()
@click.argument("dataset_paths", type=str, nargs=-1)
@click.option("--split", default=None, type=str, help="Split to evaluate")
def meteor(dataset_paths, split):
    baseline_index, dataset_paths = _handle_baseline_index(dataset_paths)
    scores = _meteor(dataset_paths, split)
    _print_table("METEOR", scores, dataset_paths, baseline_index)


@click.command()
@click.argument("dataset_paths", type=str, nargs=-1)
@click.option("--split", default=None, type=str, help="Split to evaluate")
def rouge(dataset_paths, split):
    baseline_index, dataset_paths = _handle_baseline_index(dataset_paths)
    scores = _rouge(dataset_paths, split)
    _print_table("ROUGE", scores, dataset_paths, baseline_index)


@click.command()
@click.argument("dataset_paths", type=str, nargs=-1)
@click.option("--split", default=None, type=str, help="Split to evaluate")
def spice(dataset_paths, split):
    baseline_index, dataset_paths = _handle_baseline_index(dataset_paths)
    scores = _spice(dataset_paths, split)
    _print_table("SPICE", scores, dataset_paths, baseline_index, spice=True)


@click.command()
@click.argument("dataset_paths", type=str, nargs=-1)
@click.option("--split", default=None, type=str, help="Split to evaluate")
def all(dataset_paths, split):
    baseline_index, dataset_paths = _handle_baseline_index(dataset_paths)
    all_scores = _pycoco_eval_cap_multi_metric([Cider, Meteor, Rouge, Spice], dataset_paths, split)
    bleu_scores = _bleu(dataset_paths, split)
    _print_bleu_scores(baseline_index, dataset_paths, bleu_scores)
    _print_table("CIDEr", all_scores[0], dataset_paths, baseline_index)
    _print_table("METEOR", all_scores[1], dataset_paths, baseline_index)
    _print_table("ROUGE", all_scores[2], dataset_paths, baseline_index)
    _print_table("SPICE", all_scores[3], dataset_paths, baseline_index, spice=True)


def _print_bleu_scores(baseline_index, dataset_paths, scores):
    table = Table(title="BLEU Scores")
    table.add_column("Dataset")
    table.add_column("BLEU @ 1")
    table.add_column("BLEU @ 2")
    table.add_column("BLEU @ 3")
    table.add_column("BLEU @ 4")

    # Direct print
    if baseline_index is None:
        for path, score in zip(dataset_paths, scores):
            table.add_row(
                os.path.basename(path),
                f"{score[0][0]:0.4f} +/- {np.std(score[1][0]):0.3f}",
                f"{score[0][1]:0.4f} +/- {np.std(score[1][1]):0.3f}",
                f"{score[0][2]:0.4f} +/- {np.std(score[1][2]):0.3f}",
                f"{score[0][3]:0.4f} +/- {np.std(score[1][3]):0.3f}",
            )
    else:
        # Print with a baseline
        for i, (path, score) in enumerate(zip(dataset_paths, scores)):
            if i == baseline_index:
                table.add_row(
                    os.path.basename(path),
                    # Add relative improvement
                    f"{score[0][0]:0.4f} +/- {np.std(score[1][0]):0.3f}",
                    f"{score[0][1]:0.4f} +/- {np.std(score[1][1]):0.3f}",
                    f"{score[0][2]:0.4f} +/- {np.std(score[1][2]):0.3f}",
                    f"{score[0][3]:0.4f} +/- {np.std(score[1][3]):0.3f}",
                    style="bold",
                )
            else:
                # Handle colors for the baseline
                r1_color = "green" if score[0][0] > scores[baseline_index][0][0] else "red"
                r2_color = "green" if score[0][1] > scores[baseline_index][0][1] else "red"
                r3_color = "green" if score[0][2] > scores[baseline_index][0][2] else "red"
                r4_color = "green" if score[0][3] > scores[baseline_index][0][3] else "red"
                table.add_row(
                    os.path.basename(path),
                    # Relative score to the baseline
                    f"[{r1_color}]{score[0][0]:0.4f} +/- {np.std(score[1][0]):0.3f} ({'+' if r1_color == 'green' else '-'}{np.abs(score[0][0] - scores[baseline_index][0][0]) / (np.amax([score[0][0], scores[baseline_index][0][0]])+1e-12) * 100:0.3f}%)[/{r1_color}]",
                    f"[{r2_color}]{score[0][1]:0.4f} +/- {np.std(score[1][1]):0.3f} ({'+' if r2_color == 'green' else '-'}{np.abs(score[0][1] - scores[baseline_index][0][1]) / (np.amax([score[0][1], scores[baseline_index][0][1]])+1e-12) * 100:0.3f}%)[/{r2_color}]",
                    f"[{r3_color}]{score[0][2]:0.4f} +/- {np.std(score[1][2]):0.3f} ({'+' if r3_color == 'green' else '-'}{np.abs(score[0][2] - scores[baseline_index][0][2]) / (np.amax([score[0][2], scores[baseline_index][0][2]])+1e-12) * 100:0.3f}%)[/{r3_color}]",
                    f"[{r4_color}]{score[0][3]:0.4f} +/- {np.std(score[1][3]):0.3f} ({'+' if r4_color == 'green' else '-'}{np.abs(score[0][3] - scores[baseline_index][0][3]) / (np.amax([score[0][3], scores[baseline_index][0][3]])+1e-12) * 100:0.3f}%)[/{r4_color}]",
                )

    rich.print(table)


# BLEU is annoying, since it's special :)
@click.command()
@click.argument("dataset_paths", type=str, nargs=-1)
@click.option("--split", default=None, type=str, help="Split to evaluate")
def bleu(dataset_paths, split):
    # Handle baseline index
    baseline_index, dataset_paths = _handle_baseline_index(dataset_paths)
    scores = _bleu(dataset_paths, split)

    # Rich print scores
    _print_bleu_scores(baseline_index, dataset_paths, scores)


score.add_command(ciderd)
score.add_command(meteor)
score.add_command(bleu)
score.add_command(rouge)
score.add_command(spice)
score.add_command(all)
