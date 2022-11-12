# import copy
# import math
# import os
# import pickle
# from collections import defaultdict
from functools import lru_cache

from vdtk.metrics.cider.cider import SingleScoreCider

from . import DistanceFunction
from .normalize import coco_normalize


class CIDERDDistance(DistanceFunction):
    def __init__(
        self,
    ) -> None:
        self.scorer = SingleScoreCider()

    @lru_cache(None)
    def __call__(self, x: str, y: str) -> float:
        return 10 - self.scorer.compute_score([coco_normalize(x)], [coco_normalize(y)])


class CIDERDScore(DistanceFunction):
    def __init__(
        self,
    ) -> None:
        self.scorer = SingleScoreCider()

    @lru_cache(None)
    def __call__(self, x: str, y: str) -> float:
        sc = self.scorer.compute_score([coco_normalize(x)], [coco_normalize(y)])
        return sc
