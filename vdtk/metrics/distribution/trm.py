import logging
import random
from typing import Any, Dict, Optional, Sequence, Type

import numpy as np

from .distance import DistanceFunction
from .scorer import MetricScorer


class TriangleRankMetricScorer(MetricScorer):
    def __init__(
        self,
        distance_function: Type[DistanceFunction],
        num_uk_samples: int = 500,
        num_null_samples: int = 50,
        num_workers: Optional[int] = None,
        log_p: bool = False,
        maintain_worker_state: bool = True,
        quiet: bool = False,
        supersample: bool = False,
    ) -> None:

        super().__init__(
            num_null_samples,
            num_workers,
            log_p,
            maintain_worker_state,
            quiet,
            supersample,
        )
        self._distance_function = distance_function
        self._num_uk_samples = num_uk_samples

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
        return self._compute_uk(candidates, references, worker_state["distance"]) + self._compute_uk(
            references, candidates, worker_state["distance"]
        )

    def _indicator(self, x: str, y_1: str, y_2: str, k: int, distance: DistanceFunction) -> int:
        deltas = {"dy1y2": distance(y_1, y_2), "dxy1": distance(x, y_1), "dxy2": distance(x, y_2)}
        dsorted = sorted(deltas.items(), key=lambda x: x[1])

        if k == 1:
            if dsorted[0][0] == "dy1y2":
                return 1 if dsorted[0][1] != dsorted[1][1] else np.random.choice([0, 1], p=[2 / 3, 1 / 3])
            if dsorted[0][1] == dsorted[1][1]:
                if dsorted[1][0] == "dy1y2":
                    return np.random.choice([0, 1], p=[2 / 3, 1 / 3])
                if dsorted[1][1] == dsorted[2][1]:
                    return np.random.choice([0, 1], p=[2 / 3, 1 / 3])
            return 0
        elif k == 2:
            if dsorted[1][0] == "dy1y2":
                return (
                    1
                    if dsorted[1][1] not in [dsorted[2][1], dsorted[0][1]]
                    else np.random.choice([0, 1], p=[2 / 3, 1 / 3])
                )

            if dsorted[1][1] == dsorted[0][1] and dsorted[0][0] == "dy1y2":
                return np.random.choice([0, 1], p=[2 / 3, 1 / 3])
            if dsorted[1][1] == dsorted[2][1] and dsorted[2][0] == "dy1y2":
                return np.random.choice([0, 1], p=[2 / 3, 1 / 3])
            return 0
        elif k == 3:
            if dsorted[2][0] == "dy1y2":
                return 1 if dsorted[2][1] != dsorted[1][1] else np.random.choice([0, 1], p=[2 / 3, 1 / 3])
            if dsorted[2][1] == dsorted[1][1] == dsorted[0][1]:
                return np.random.choice([0, 1], p=[2 / 3, 1 / 3])
            return 0
        return np.random.choice([0, 1], p=[2 / 3, 1 / 3])

    def _compute_uk(self, candidates: Sequence[str], references: Sequence[str], distance: DistanceFunction) -> float:
        ukdev = 0
        for k in range(3):
            acc = 0
            for _ in range(self._num_uk_samples):
                # Sample a set of values, and compute the indicator
                candidate = random.choice(candidates)
                reference_a, reference_b = random.sample(references, 2)
                acc += self._indicator(candidate, reference_a, reference_b, k=k, distance=distance)
            ukdev += np.abs(acc / self._num_uk_samples - 1 / 3)
        return ukdev
