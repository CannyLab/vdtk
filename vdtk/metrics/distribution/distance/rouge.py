from functools import lru_cache

from rouge_score import rouge_scorer

from .normalize import coco_normalize

from . import DistanceFunction


class ROUGELDistance(DistanceFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scorer = rouge_scorer.RougeScorer(["rougeL"])

    @lru_cache(None)
    def __call__(self, x: str, y: str) -> float:
        return 1 - self.scorer.score(coco_normalize(x), coco_normalize(y))["rougeL"].fmeasure
