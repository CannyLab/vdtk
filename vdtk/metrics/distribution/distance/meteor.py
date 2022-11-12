from functools import lru_cache

import nltk
from nltk.translate.meteor_score import single_meteor_score

from . import DistanceFunction
from .normalize import coco_normalize


class MeteorDistance(DistanceFunction):
    def __init__(self) -> None:
        nltk.download("omw-1.4", quiet=True)

    @lru_cache(None)
    def __call__(self, x: str, y: str) -> float:
        return 1 - single_meteor_score(coco_normalize(x).split(), coco_normalize(y).split())
