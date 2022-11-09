import logging
from typing import Any, Dict, Optional, Sequence, Type

import ot
import torch

from vdtk.metrics.distribution.distance import DistanceFunction
from vdtk.metrics.distribution.scorer import MetricScorer


def compute_emd(x: Sequence[str], y: Sequence[str], distance: DistanceFunction) -> float:

    if len(x) != len(y):
        x = x[: min(len(x), len(y))]
        y = y[: min(len(x), len(y))]

    # Compute the pairwise distance matrix
    M = torch.zeros(len(x), len(y))
    for i, sentence_a in enumerate(x):
        for j, sentence_b in enumerate(y):
            M[i, j] = distance(sentence_a, sentence_b)

    M /= M.max()
    a, b = torch.ones((len(x),)) / len(x), torch.ones((len(y),)) / len(y)  # Uniform distributions on A, B
    return ot.emd2(a, b, M).item()


class PointwiseEMDMetricScorer(MetricScorer):
    def __init__(
        self,
        distance_function: Type[DistanceFunction],
        num_null_samples: int = 50,
        num_workers: Optional[int] = None,
        log_p: bool = False,
    ) -> None:

        super().__init__(num_null_samples, num_workers, log_p)
        self._distance_function = distance_function

    def _initialize_worker_state(self) -> Dict[str, Any]:
        return {
            "distance": self._distance_function(),
        }

    def _score(
        self,
        candidates: Sequence[str],
        references: Sequence[str],
        worker_state: Dict[str, Any],
    ) -> Optional[float]:
        if len(candidates) < 2 or len(references) < 2:
            logging.warning(f"Candidates or references are too short: {len(candidates)} {len(references)}")
            return None
        return compute_emd(candidates, references, distance=worker_state["distance"])
