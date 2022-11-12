from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rich.progress import track

NGram = Tuple[Optional[str], ...]


def _dd1() -> float:
    return 0.0


def _dd0() -> Dict[Optional[str], float]:
    return defaultdict(_dd1)


def find_ngrams(input_list: Sequence[Optional[str]], n: int) -> List[NGram]:
    # Pad the list with None values to the required n
    input_list = [None] * (n - 1) + list(input_list) + [None] * (n - 1)
    return zip(*[input_list[i:] for i in range(n)])  # type: ignore


class NGramLM:
    def __init__(self, samples: Sequence[Sequence[Optional[str]]], n: int = 2):
        self._model: Dict[NGram, Dict[Optional[str], float]] = defaultdict(_dd0)
        self._count = 0
        self._n = n

        for sample in track(samples, transient=True, description="Training"):
            for ngram in find_ngrams(sample, n):
                self._model[ngram[:-1]][ngram[-1]] += 1
                self._count += 1

        # Build likelihoods
        for gram in self._model:
            total_count = float(sum(self._model[gram].values()))
            for w3 in self._model[gram]:
                self._model[gram][w3] /= total_count

    @property
    def count(self) -> int:
        return self._count

    @property
    def model(self) -> Dict[NGram, Dict[Optional[str], float]]:
        return self._model

    def log_likelihood(self, sample: List[str]) -> float:
        log_likelihood = 0.0
        ngms = list(find_ngrams(sample, self._n))
        for ngram in ngms:
            log_likelihood += max(np.log(self._model[ngram[:-1]][ngram[-1]]), -12)
        return log_likelihood / (len(ngms) + 1e-13)
