import itertools
import logging
import random
from collections import defaultdict
from typing import Optional

import click
import numpy as np
import rich
from rich.progress import track

from vdtk.data_utils import load_dataset
from vdtk.metrics.bleu.bleu import Bleu
from vdtk.stats_utils import descr


@click.command()
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option("--split", default=None, type=str, help="Split to evaluate")
@click.option("--samples", default=1, type=int, help="The number of samples to get")
@click.option("--candidates", default=False, is_flag=True, help="Evaluate candidates instead of references")
def qualitative_sample(
    dataset_path: str, split: Optional[str] = None, samples: int = 1, candidates: bool = False
) -> None:

    logging.info("Loading dataset...")
    data = load_dataset(dataset_path)
    if split is not None:
        # Filter the data for the correct split
        data = [s for s in data if s.split == split]
    # Filter data for samples with references
    data = [s for s in data if (s.references if not candidates else s.candidates)]

    console = rich.console.Console()
    with console.capture() as capture:
        for _ in track(list(range(samples)), description="Computing stats...", transient=True):

            # Randomly sample a single element from the dataset
            sample = random.choice(data)

            # Compute the within-sample BERT embedding distances
            sample_distances = {}
            for cp_a, emb_a in zip(
                (sample.references if not candidates else sample.candidates),
                (sample.reference_embeddings if not candidates else sample.candidate_embeddings),
            ):
                sample_distances[cp_a] = {}
                for cp_b, emb_b in zip(
                    (sample.references if not candidates else sample.candidates),
                    (sample.reference_embeddings if not candidates else sample.candidate_embeddings),
                ):
                    if cp_a != cp_b:
                        sample_distances[cp_a][cp_b] = 1 - np.dot(emb_a, emb_b) / (
                            np.linalg.norm(emb_a) * np.linalg.norm(emb_b)
                        )

            # Describe the inter-sample distances
            dff = list(
                itertools.chain.from_iterable(
                    (sample_distances[cp_a][cp_b] for cp_b in sample_distances[cp_a]) for cp_a in sample_distances
                )
            )
            sample_distances_descr = descr(dff)

            # Compute the mean embedding, and distance to the mean
            mean_embedding = np.mean(
                (sample.reference_embeddings if not candidates else sample.candidate_embeddings), axis=0
            )
            mean_distances = {}
            for cp_a, emb_a in zip(
                (sample.references if not candidates else sample.candidates),
                (sample.reference_embeddings if not candidates else sample.candidate_embeddings),
            ):
                mean_distances[cp_a] = 1 - np.dot(emb_a, mean_embedding) / (
                    np.linalg.norm(emb_a) * np.linalg.norm(mean_embedding)
                )

            # Compute the inter-sample leave-one-out BLEU scores
            scorer = Bleu(4)
            bleu_scores = defaultdict(dict)
            for c, cp_a in zip(
                (sample.references if not candidates else sample.candidates),
                (sample.references_tokenized_text if not candidates else sample.candidates_tokenized_text),
            ):
                hypothesis = {0: [" ".join(cp_a)]}
                for cb, cp_b in zip(
                    (sample.references if not candidates else sample.candidates),
                    (sample.references_tokenized_text if not candidates else sample.candidates_tokenized_text),
                ):
                    if tuple(cp_a) != tuple(cp_b):
                        bleu_scores[c][cb] = scorer.compute_score({0: [" ".join(cp_b)]}, hypothesis)[0][-1]

            # Get the best mean caption based on BLEU
            mabs = {k: np.amax([v for _, v in vals.items()]) for k, vals in bleu_scores.items()}
            best_bleu_mean_caption = max(mabs.items(), key=lambda x: x[1])

            # Get the best mean caption based on mean embedding distance
            best_mean_embedding_caption = min(mean_distances.items(), key=lambda x: x[1])
            meds = descr(list(mean_distances.values()))

            console.print()
            console.rule(f"[bold]Sample: {sample._id}")
            for idx, elem in enumerate(sorted(mean_distances, key=mean_distances.get)):  # type: ignore
                fmt_string = f"- {elem} (Dist: {mean_distances[elem]:.2f}) (BLEU@4: {np.amax(list(bleu_scores[elem].values())):.4f})"
                if idx == 0 and elem == best_bleu_mean_caption[0]:
                    console.print(fmt_string, style="green")
                elif elem == best_bleu_mean_caption[0]:
                    console.print(fmt_string, style="red")
                elif idx == 0:
                    console.print(fmt_string, style="yellow")
                else:
                    console.print(fmt_string)
            console.print()
            console.print(
                f"Mean Leave One Out BLEU@4 score: {np.mean([np.amax(list(v.values())) for v in bleu_scores.values()]):.4f}"
            )
            console.print(
                f"Within-Sample mean BERT Embedding distances [Min, Mean, Max]: [{meds['min']:.2f}, {meds['mean']:.2f}, {meds['max']:.2f}]"
            )

    print(capture.get())
