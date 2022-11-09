from functools import lru_cache

import nltk

from .normalize import coco_normalize

from . import DistanceFunction


class MeteorDistance(DistanceFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        nltk.download("omw-1.4", quiet=True)

    @lru_cache(None)
    def __call__(self, x: str, y: str) -> float:
        return 1 - nltk.translate.meteor_score.single_meteor_score(coco_normalize(x).split(), coco_normalize(y).split())  # type: ignore
