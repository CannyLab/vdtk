import os
from functools import lru_cache
from typing import Any, List, Optional, Tuple

import click
import clip
import numpy as np
import rich
import torch
from PIL import Image
from rich.progress import track
from rich.table import Table

from vdtk.data_utils import Sample, load_dataset
from vdtk.score import _handle_baseline_index
from vdtk.utils.rich import baseline_column

Result = List[
    Tuple[
        Tuple[List[np.floating], List[np.floating], List[np.floating], List[np.floating], List[np.floating]],
        Tuple[List[np.floating], List[np.floating], List[np.floating], List[np.floating], List[np.floating]],
    ],
]


@lru_cache
def clip_model() -> Tuple[Any, Any, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


@lru_cache
def _get_feature(media_path: str) -> torch.Tensor:
    model, preprocess, device = clip_model()
    image = preprocess(Image.open(media_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.reshape(-1)


def _get_image_feature_db(data: List[Sample]) -> torch.Tensor:
    features = []
    for sample in track(data, description="Featurizing dataset", transient=True):
        features.append(_get_feature(sample.media_path))
    return torch.stack(features).to("cpu" if not torch.cuda.is_available() else "cuda")


def _get_text_features(
    sample: Sample, text_features: torch.Tensor, char_limit: int = 300
) -> Tuple[torch.Tensor, torch.Tensor]:
    model, _, device = clip_model()
    candidate_text = clip.tokenize([i[:char_limit] for i in sample.candidates]).to(device)
    reference_text = clip.tokenize([i[:char_limit] for i in sample.references]).to(device)
    with torch.no_grad():
        candidate_text_features = model.encode_text(candidate_text)
        reference_text_features = model.encode_text(reference_text)
        candidate_text_features /= candidate_text_features.norm(dim=-1, keepdim=True)
        reference_text_features /= reference_text_features.norm(dim=-1, keepdim=True)

    return candidate_text_features, reference_text_features


def _add_table_row(
    i: int,
    baseline_index: Optional[int],
    table: Table,
    name: str,
    scores: np.ndarray,
    outputs: Result,
    is_candidate: bool,
) -> None:
    (rank, rrank, recall_1, recall_5, recall_max) = scores

    if i is None or baseline_index is None:
        table.add_row(
            name,
            f"{np.mean(rank):.4f} ± {np.std(rank):.4f}",
            f"{np.mean(rrank):.4f} ± {np.std(rrank):.4f}",
            f"{np.mean(recall_1):.4f} ± {np.std(recall_1):.4f}",
            f"{np.mean(recall_5):.4f} ± {np.std(recall_5):.4f}",
            f"{np.amax(recall_max):.4f}",
        )
    else:
        if i == baseline_index and not is_candidate:
            table.add_row(
                name,
                f"{np.mean(rank):.4f} ± {np.std(rank):.4f}",
                f"{np.mean(rrank):.4f} ± {np.std(rrank):.4f}",
                f"{np.mean(recall_1):.4f} ± {np.std(recall_1):.4f}",
                f"{np.mean(recall_5):.4f} ± {np.std(recall_5):.4f}",
                f"{np.amax(recall_max):.4f}",
            )
        else:
            table.add_row(
                name,
                baseline_column(rank, outputs[baseline_index][1][0], positive=False),  # type: ignore
                baseline_column(rrank, outputs[baseline_index][1][1]),  # type: ignore
                baseline_column(recall_1, outputs[baseline_index][1][2]),  # type: ignore
                baseline_column(recall_5, outputs[baseline_index][1][3]),  # type: ignore
                baseline_column(
                    recall_max,
                    outputs[baseline_index][1][4],  # type: ignore
                    aggregate=np.amax,
                    baseline_aggregate=np.amax,
                    positive=False,
                ),
            )


@click.command()
@click.argument("dataset_paths", type=str, nargs=-1)
@click.option("--split", default=None, type=str, help="Split to evaluate")
@click.option("--media-root", default=None, type=str, help="Root directory for media")
def clip_recall(
    dataset_paths: List[str],
    split: Optional[str] = None,
    media_root: Optional[str] = None,
) -> None:

    # Get the baseline
    baseline_index, dataset_paths = _handle_baseline_index(dataset_paths)

    outputs: List[Result] = []
    for ds in dataset_paths:
        data = load_dataset(ds, media_root)
        if split is not None:
            # Filter the data for the correct split
            data = [s for s in data if s.split == split]

        # Compute the features
        image_feature_db = _get_image_feature_db(data)

        # Compute the recall
        candidate_scores = []
        reference_scores = []
        for index, sample in enumerate(
            track(data, description=f"Computing recall for dataset {os.path.basename(ds)}", transient=True)
        ):
            candidate_features, reference_features = _get_text_features(sample, image_feature_db)
            candidate_similarity_scores = image_feature_db @ candidate_features.T
            candidate_ranks = (candidate_similarity_scores > candidate_similarity_scores[index]).sum(dim=0)

            reference_similarity_scores = image_feature_db @ reference_features.T
            reference_ranks = (reference_similarity_scores > reference_similarity_scores[index]).sum(dim=0)

            candidate_scores.append((candidate_ranks + 1).cpu().numpy())
            reference_scores.append((reference_ranks + 1).cpu().numpy())

        outputs.append(
            (
                (
                    # rank
                    [np.mean(i) for i in candidate_scores],
                    # Reciprocal rank
                    [np.mean(1 / i) for i in candidate_scores],
                    # Recall at 1
                    [np.mean(i <= 1) for i in candidate_scores],
                    # Recall at 5
                    [np.mean(i <= 5) for i in candidate_scores],
                    # 100% recall at
                    [np.amax(i) for i in candidate_scores],  # type: ignore
                ),
                (
                    # rank
                    [np.mean(i) for i in reference_scores],
                    # Reciprocal rank
                    [np.mean(1 / i) for i in reference_scores],
                    # Recall at 1
                    [np.mean(i <= 1) for i in reference_scores],
                    # Recall at 5
                    [np.mean(i <= 5) for i in reference_scores],
                    # 100% recall at
                    [np.amax(i) for i in reference_scores],
                ),
            )
        )

    # Print the results
    table = Table(title=f"CLIP Recall")
    table.add_column("Dataset", justify="left", style="cyan", no_wrap=True)
    table.add_column("Mean Rank", justify="right", style="magenta")
    table.add_column("Mean Reciprocal Rank", justify="right", style="magenta")
    table.add_column("Recall @ 1", justify="right", style="magenta")
    table.add_column("Recall @ 5", justify="right", style="magenta")
    table.add_column("100% Recall", justify="right", style="magenta")
    for i, (ds, (candidate_scores, reference_scores)) in enumerate(zip(dataset_paths, outputs)):  # type: ignore
        # Add The candidate scores
        _add_table_row(
            i,
            baseline_index,
            table,
            os.path.basename(ds) + " (candidate)",
            candidate_scores,  # type: ignore
            outputs,  # type: ignore
            True,
        )
        _add_table_row(
            i,
            baseline_index,
            table,
            os.path.basename(ds) + " (reference)",
            reference_scores,  # type: ignore
            outputs,  # type: ignore
            False,
        )
    rich.print(table)
