import os
from functools import lru_cache

try:
    from bleurt import score
except ImportError:
    score = None

from . import DistanceFunction


class BLEURTDistance(DistanceFunction):
    def __init__(
        self,
    ):
        if score is None:
            raise ImportError("BLEURT is not installed. Please install it with " "`pip install bleurt`.")

        super().__init__()
        self._scorer = score.BleurtScorer(checkpoint=os.path.expanduser("~/.cache/bleurt/BLEURT-20"))

    @lru_cache(None)
    def __call__(self, x: str, y: str) -> float:
        return 2 / (1e-12 + 1 + self._scorer.score(candidates=[x], references=[y])[0])
