from collections import defaultdict
from typing import List, Sequence

import numpy as np
import tqdm


def _dd1():
    return 0


def _dd0():
    return defaultdict(_dd1)


def find_ngrams(input_list, n):
    # Pad the list with None values to the required n
    input_list = [None] * (n - 1) + input_list + [None] * (n - 1)
    return zip(*[input_list[i:] for i in range(n)])


class NGramLM:
    def __init__(self, samples: Sequence[Sequence[str]], n: int = 2):
        self._model = defaultdict(_dd0)
        self._count = 0
        self._n = n

        for sample in tqdm.tqdm(samples, leave=False):
            for ngram in find_ngrams(sample, n):
                self._model[ngram[:-1]][ngram[-1]] += 1
                self._count += 1

        # Build likelihoods
        for gram in self._model:
            total_count = float(sum(self._model[gram].values()))
            for w3 in self._model[gram]:
                self._model[gram][w3] /= total_count

    @property
    def count(self):
        return self._count

    @property
    def model(self):
        return self._model

    def log_likelihood(self, sample: List[str]) -> float:
        log_likelihood = 0.0
        ngms = list(find_ngrams(sample, self._n))
        for ngram in ngms:
            log_likelihood += max(np.log(self._model[ngram[:-1]][ngram[-1]]), -12)
        return log_likelihood / (len(ngms) + 1e-13)
