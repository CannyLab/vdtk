import itertools
import json
import os
import warnings
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union

import click
import numpy as np
import rich
import torch
from bert_score import BERTScorer

with warnings.catch_warnings():
    # Ignore the warnings about AVX512 missing on Mauve
    warnings.simplefilter("ignore")
    import mauve

try:
    from bleurt import score as bleurt_score
except ImportError as e:
    bleurt_score = None

from rich.progress import track
from rich.table import Table

# from vdtk.metrics.bleu.bleu import Bleu
from vdtk.metrics.bleu.bleu import Bleu
from vdtk.metrics.cider.cider import CiderBase as Cider
from vdtk.metrics.distribution import (MetricScorer, MMDBertMetricScorer,
                                       MMDCLIPMetricScorer,
                                       MMDFastTextMetricScorer,
                                       MMDGloveMetricScorer,
                                       TriangleRankMetricScorer)
from vdtk.metrics.distribution.distance import (BERTDistance,
                                                BERTScoreDistance,
                                                BLEU4Distance, BLEURTDistance,
                                                CIDERDDistance,
                                                DistanceFunction,
                                                MeteorDistance, ROUGELDistance)
from vdtk.metrics.meteor.meteor import MeteorBase as Meteor
from vdtk.metrics.rouge.rouge import RougeBase as Rouge
from vdtk.metrics.spice.spice import Spice
from vdtk.metrics.tokenizer.ptbtokenizer import PTBTokenizer

MMDMetricScorer = Union[
    Type[MMDBertMetricScorer], Type[MMDCLIPMetricScorer], Type[MMDFastTextMetricScorer], Type[MMDGloveMetricScorer]
]


@click.group()
def score() -> None:
    pass


def _distribution_metric(
    scorer: MetricScorer,
    dataset_paths: Sequence[str],
    split: Optional[str],
) -> List[Tuple[float, List[float]]]:
    tokenizer = PTBTokenizer()

    # Load the paths
    outputs = []
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

        scores = scorer(candidates, references)
        flat_scores = [s.test_statistic for s in scores.values() if s is not None]
        total_score = float(np.mean(flat_scores))
        outputs.append((total_score, flat_scores))

    return outputs


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
    scores: List[List[Tuple[float, List[float]]]] = [[] for _ in measure]
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
) -> List[Tuple[Tuple[float, float, float, float], Tuple[List[float], List[float], List[float], List[float]]]]:
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

        avg_scores, score = Bleu().compute_score_flat(references, candidates)

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


def _bleurt(dataset_paths: Sequence[str], split: Optional[str] = None) -> List[Tuple[float, List[float]]]:

    if bleurt_score is None:
        raise ImportError("BLEURT requires the bleurt package to be installed.")
    assert bleurt_score is not None

    scores = []
    for path in dataset_paths:
        # Load the candidates and references from the dataset
        with open(path, "r") as f:
            data = json.load(f)
            candidates = [
                sample["candidates"]
                for i, sample in enumerate(data)
                if sample.get("split", None) == split or split is None
            ]
            references = [
                sample["references"]
                for i, sample in enumerate(data)
                if sample.get("split", None) == split or split is None
            ]

            scorer = bleurt_score.BleurtScorer()
            sample_scores = []
            for c, r in track(list(zip(candidates, references)), transient=True):
                candidate_scores = []
                for cn in c:
                    candidate_scores.append(np.max(scorer.score(candidates=[cn for _ in r], references=r)))
                sample_scores.append(np.mean(candidate_scores))

            scores.append((np.mean(sample_scores), sample_scores))

    return scores


def _bert_score(dataset_paths: Sequence[str], split: Optional[str] = None) -> List[Tuple[float, List[float]]]:
    scores = []
    for path in dataset_paths:
        # Load the candidates and references from the dataset
        with open(path, "r") as f:
            data = json.load(f)
            candidates = [
                sample["candidates"]
                for i, sample in enumerate(data)
                if sample.get("split", None) == split or split is None
            ]
            references = [
                sample["references"]
                for i, sample in enumerate(data)
                if sample.get("split", None) == split or split is None
            ]

            sample_scores = []
            scorer = BERTScorer(lang="en", rescale_with_baseline=True)
            for c, r in track(list(zip(candidates, references)), transient=True):
                candidate_scores = []
                for cn in c:
                    _, _, F = scorer.score([cn], [r])
                    assert isinstance(F, torch.Tensor)
                    candidate_scores.append(F.cpu().item())
                sample_scores.append(float(np.mean(candidate_scores)))

            scores.append((float(np.mean(sample_scores)), sample_scores))

    return scores


def _mauve(dataset_paths: Sequence[str], split: Optional[str] = None) -> List[Tuple[float, List[float]]]:
    scores = []
    for path in dataset_paths:
        # Load the candidates and references from the dataset
        with open(path, "r") as f:
            data = json.load(f)
            candidates = [
                sample["candidates"]
                for i, sample in enumerate(data)
                if sample.get("split", None) == split or split is None
            ]
            references = [
                sample["references"]
                for i, sample in enumerate(data)
                if sample.get("split", None) == split or split is None
            ]

            all_candidates = list(itertools.chain.from_iterable(candidates))
            all_references = list(itertools.chain.from_iterable(references))

            score = mauve.compute_mauve(
                p_text=all_candidates,
                q_text=all_references,
                device_id=0 if torch.cuda.is_available() else -1,
                verbose=False,
            )
            scores.append((score.mauve, [score.mauve]))

    return scores


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
    swap_colors: bool = False,
) -> None:

    if spice:
        # There are some issues with the Spice score
        spice_raw = [[k["All"]["f"] for k in s[1]] for s in scores]  # type: ignore
        scores = [(s[0], k) for s, k in zip(scores, spice_raw)]

    table = Table(title=f"{label} Scores")
    table.add_column("Dataset")
    table.add_column(f"{label}")
    table.add_column(f"Max {label}")
    table.add_column(f"Min {label}")

    good_color = "green" if not swap_colors else "red"
    bad_color = "red" if not swap_colors else "green"

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
                r1_color = good_color if score[0] > scores[baseline_index][0] else bad_color
                r2_color = good_color if np.max(score[1]) > np.max(scores[baseline_index][1]) else bad_color
                r3_color = good_color if np.min(score[1]) > np.min(scores[baseline_index][1]) else bad_color
                table.add_row(
                    os.path.basename(path),
                    # Relative score to the baseline
                    f"[{r1_color}]{score[0]:0.4f} +/- {np.std(score[1]):0.3f} ({'+' if r1_color == good_color else '-'}{np.abs(score[0] - scores[baseline_index][0]) / (np.amax([score[0], scores[baseline_index][0]])+1e-12) * 100:0.3f}%)[/{r1_color}]",  # noqa: E501
                    f"[{r2_color}]{np.max(score[1]):0.4f} ({'+' if r2_color == good_color else '-'}{np.abs(np.max(score[1]) - np.max(scores[baseline_index][1])) / (np.amax([np.max(score[1]), np.max(scores[baseline_index][1])])+1e-12)*100:0.3f}%)[/{r2_color}]",  # noqa: E501
                    f"[{r3_color}]{np.min(score[1]):0.4f} ({'+' if r3_color == good_color else '-'}{np.abs(np.min(score[1]) - np.min(scores[baseline_index][1])) / (np.amax([np.min(score[1]), np.min(scores[baseline_index][1])])+1e-12)*100:0.3f}%)[/{r3_color}]",  # noqa: E501
                )
    rich.print(table)


def _print_bleu_scores(
    baseline_index: Optional[int],
    dataset_paths: List[str],
    scores: List[Tuple[Tuple[float, float, float, float], Tuple[List[float], List[float], List[float], List[float]]]],
) -> None:
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
                    f"[{r1_color}]{score[0][0]:0.4f} +/- {np.std(score[1][0]):0.3f} ({'+' if r1_color == 'green' else '-'}{np.abs(score[0][0] - scores[baseline_index][0][0]) / (np.amax([score[0][0], scores[baseline_index][0][0]])+1e-12) * 100:0.3f}%)[/{r1_color}]",  # noqa: E501
                    f"[{r2_color}]{score[0][1]:0.4f} +/- {np.std(score[1][1]):0.3f} ({'+' if r2_color == 'green' else '-'}{np.abs(score[0][1] - scores[baseline_index][0][1]) / (np.amax([score[0][1], scores[baseline_index][0][1]])+1e-12) * 100:0.3f}%)[/{r2_color}]",  # noqa: E501
                    f"[{r3_color}]{score[0][2]:0.4f} +/- {np.std(score[1][2]):0.3f} ({'+' if r3_color == 'green' else '-'}{np.abs(score[0][2] - scores[baseline_index][0][2]) / (np.amax([score[0][2], scores[baseline_index][0][2]])+1e-12) * 100:0.3f}%)[/{r3_color}]",  # noqa: E501
                    f"[{r4_color}]{score[0][3]:0.4f} +/- {np.std(score[1][3]):0.3f} ({'+' if r4_color == 'green' else '-'}{np.abs(score[0][3] - scores[baseline_index][0][3]) / (np.amax([score[0][3], scores[baseline_index][0][3]])+1e-12) * 100:0.3f}%)[/{r4_color}]",  # noqa: E501
                )

    rich.print(table)


def _simple_function_builder(
    metric: Callable[[List[str], Optional[str]], Any], function_name: str, string: str
) -> Tuple[Callable[[List[str], Optional[str]], None], click.Command]:
    def _metric_function(dataset_paths: List[str], split: Optional[str]) -> None:
        baseline_index, dataset_paths = _handle_baseline_index(dataset_paths)
        scores = metric(dataset_paths, split)
        _print_table(string, scores, dataset_paths, baseline_index, spice=(metric == _spice))

    return _metric_function, click.command(name=function_name)(
        click.argument("dataset_paths", type=str, nargs=-1)(
            click.option("--split", default=None, type=str, help="Split to evaluate")(_metric_function)
        )
    )


# TODO: Support multiple workers
# TODO: Support custom REPR for functions
def _trm_function_builder(
    distance_function: Type[DistanceFunction], function_name: str, string: str
) -> Tuple[Callable[[List[str], Optional[str], bool, int], None], click.Command]:
    def _metric_function(dataset_paths: List[str], split: Optional[str], supersample: bool, num_uk_samples: int = 500) -> None:
        baseline_index, dataset_paths = _handle_baseline_index(dataset_paths)
        scorer = TriangleRankMetricScorer(
            distance_function=distance_function,
            num_uk_samples=num_uk_samples,
            num_null_samples=0,
            supersample=supersample,
        )
        scores = _distribution_metric(scorer, dataset_paths, split)
        _print_table(string, scores, dataset_paths, baseline_index, swap_colors=True)

    return _metric_function, click.command(name=function_name)(
        click.argument("dataset_paths", type=str, nargs=-1)(
            click.option("--split", default=None, type=str, help="Split to evaluate")(
                click.option(
                    "--supersample",
                    default=False,
                    is_flag=True,
                    type=bool,
                    help="Supersample the number of samples to compute the metric",
                )(
                    click.option(
                        "--num_uk_samples",
                        default=500,
                        type=int,
                        help="Number of unknown samples to use for the metric",
                    )(_metric_function)
                )
            )
        )
    )


# TODO: Support multiple workers
# TODO: Support custom REPR for functions
def _mmd_function_builder(
    metric_scorer_class: MMDMetricScorer, function_name: str, string: str
) -> Tuple[Callable[[List[str], Optional[str], bool, Optional[float]], None], click.Command]:
    def _metric_function(dataset_paths: List[str], split: Optional[str], supersample: bool, mmd_sigma: Optional[float]) -> None:
        baseline_index, dataset_paths = _handle_baseline_index(dataset_paths)
        scorer = metric_scorer_class(
            mmd_sigma if mmd_sigma is not None else "median", num_null_samples=0, supersample=supersample
        )
        scores = _distribution_metric(scorer, dataset_paths, split)
        _print_table(string, scores, dataset_paths, baseline_index, swap_colors=True)

    return _metric_function, click.command(name=function_name)(
        click.argument("dataset_paths", type=str, nargs=-1)(
            click.option("--split", default=None, type=str, help="Split to evaluate")(
                click.option(
                    "--supersample",
                    default=False,
                    is_flag=True,
                    type=bool,
                    help="Supersample the number of samples to compute the metric",
                )(click.option("--mmd-sigma", default=None, type=float, help="MMD sigma")(_metric_function))
            )
        )
    )


ciderd, ciderd_command = _simple_function_builder(_ciderd, "ciderd", "CIDEr-D")
meteor, meteor_command = _simple_function_builder(_meteor, "meteor", "METEOR")
rouge, rouge_command = _simple_function_builder(_rouge, "rouge", "ROUGE")
spice, spice_command = _simple_function_builder(_spice, "spice", "SPICE")
bleurt, bleurt_command = _simple_function_builder(_bleurt, "bleurt", "BLEURT")
bert_score, bert_score_command = _simple_function_builder(_bert_score, "bert-score", "BERTScore")
mauve_score, mauve_score_command = _simple_function_builder(_mauve, "mauve", "Mauve")


trm_bleu, trm_bleu_command = _trm_function_builder(BLEU4Distance, "trm-bleu", "TRM-BLEU")
trm_cider, trm_cider_command = _trm_function_builder(CIDERDDistance, "trm-cider", "TRM-CIDEr")
trm_meteor, trm_meteor_command = _trm_function_builder(MeteorDistance, "trm-meteor", "TRM-METEOR")
trm_rouge, trm_rouge_command = _trm_function_builder(ROUGELDistance, "trm-rouge", "TRM-ROUGE")
trm_bleurt, trm_bleurt_command = _trm_function_builder(BLEURTDistance, "trm-bleurt", "TRM-BLEURT")
trm_bert_score, trm_bert_score_command = _trm_function_builder(BERTScoreDistance, "trm-bert-score", "TRM-BERTScore")
trm_bert, trm_bert_command = _trm_function_builder(BERTDistance, "trm-bert", "TRM-BERT")

mmd_bert, mmd_bert_command = _mmd_function_builder(MMDBertMetricScorer, "mmd-bert", "MMD-BERT")
mmd_clip, mmd_clip_command = _mmd_function_builder(MMDCLIPMetricScorer, "mmd-clip", "MMD-CLIP")
mmd_fasttext, mmd_fasttext_command = _mmd_function_builder(MMDFastTextMetricScorer, "mmd-fasttext", "MMD-FastText")
mmd_glove, mmd_glove_command = _mmd_function_builder(MMDGloveMetricScorer, "mmd-glove", "MMD-GloVe")


# BLEU is annoying, since it's special :)
@click.command()
@click.argument("dataset_paths", type=str, nargs=-1)
@click.option("--split", default=None, type=str, help="Split to evaluate")
def bleu(dataset_paths: List[str], split: str) -> None:
    # Handle baseline index
    baseline_index, dataset_paths = _handle_baseline_index(dataset_paths)
    scores = _bleu(dataset_paths, split)

    # Rich print scores
    _print_bleu_scores(baseline_index, dataset_paths, scores)


@click.command()
@click.argument("dataset_paths", type=str, nargs=-1)
@click.option("--split", default=None, type=str, help="Split to evaluate")
@click.option("--supersample", is_flag=True, default=False, type=bool, help="If candidates should be supersampled")
@click.option("--num_uk_samples", default=500, type=int, help="Number of unknown samples to use for the metric")
@click.option("--mmd-sigma", default=None, type=float, help="MMD sigma")
def all(
    dataset_paths: List[str],
    split: Optional[str],
    supersample: bool,
    num_uk_samples: int = 500,
    mmd_sigma: Optional[float] = None,
) -> None:

    # Simple Metrics
    baseline_index, dataset_paths_filtered = _handle_baseline_index(dataset_paths)
    all_scores = _pycoco_eval_cap_multi_metric([Cider, Meteor, Rouge, Spice], dataset_paths_filtered, split)
    bleu_scores = _bleu(dataset_paths_filtered, split)

    _print_bleu_scores(baseline_index, dataset_paths_filtered, bleu_scores)
    _print_table("CIDEr", all_scores[0], dataset_paths_filtered, baseline_index)
    _print_table("METEOR", all_scores[1], dataset_paths_filtered, baseline_index)
    _print_table("ROUGE", all_scores[2], dataset_paths_filtered, baseline_index)
    _print_table("SPICE", all_scores[3], dataset_paths_filtered, baseline_index, spice=True)
    _print_table("BLEURT", _bleurt(dataset_paths_filtered, split), dataset_paths_filtered, baseline_index)
    _print_table("BERTScore", _bert_score(dataset_paths_filtered, split), dataset_paths_filtered, baseline_index)
    _print_table("Mauve", _mauve(dataset_paths_filtered, split), dataset_paths_filtered, baseline_index)

    # MMD scores
    mmd_glove(dataset_paths, split, supersample, mmd_sigma)
    mmd_fasttext(dataset_paths, split, supersample, mmd_sigma)
    mmd_clip(dataset_paths, split, supersample, mmd_sigma)
    mmd_bert(dataset_paths, split, supersample, mmd_sigma)

    # TRM scores
    trm_bert(dataset_paths, split, supersample, num_uk_samples)
    trm_bert_score(dataset_paths, split, supersample, num_uk_samples)
    trm_bleu(dataset_paths, split, supersample, num_uk_samples)
    trm_cider(dataset_paths, split, supersample, num_uk_samples)
    trm_meteor(dataset_paths, split, supersample, num_uk_samples)
    trm_rouge(dataset_paths, split, supersample, num_uk_samples)


# Add all of the commands here...
score.add_command(ciderd_command)
score.add_command(meteor_command)
score.add_command(bleu)
score.add_command(rouge_command)
score.add_command(spice_command)
score.add_command(bleurt_command)
score.add_command(bert_score_command)
score.add_command(mauve_score_command)
score.add_command(trm_bleu_command)
score.add_command(trm_cider_command)
score.add_command(trm_meteor_command)
score.add_command(trm_rouge_command)
score.add_command(trm_bleurt_command)
score.add_command(trm_bert_command)
score.add_command(trm_bert_score_command)
score.add_command(mmd_bert_command)
score.add_command(mmd_clip_command)
score.add_command(mmd_fasttext_command)
score.add_command(mmd_glove_command)
score.add_command(all)
