class DistanceFunction:
    def __call__(self, x: str, y: str) -> float:
        raise NotImplementedError("DistanceFunction.__call__ must be implemented")


from .bert import BERTDistance, BERTScoreDistance
from .bleu import BLEU4Distance
from .cider import CIDERDDistance, CIDERDScore
from .meteor import MeteorDistance
from .rouge import ROUGELDistance
from .bleurt import BLEURTDistance
