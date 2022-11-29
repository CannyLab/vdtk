# Filename: cider.py
#
# Description: Describes the class to compute the CIDEr (Consensus-Based Image Description Evaluation) Metric
#               by Vedantam, Zitnick, and Parikh (http://arxiv.org/abs/1411.5726)
#
# Creation Date: Sun Feb  8 14:16:54 2015
#
# Authors: Ramakrishna Vedantam <vrama91@vt.edu> and Tsung-Yi Lin <tl483@cornell.edu>
import os
from typing import Any, Dict, List, Tuple, TypeVar

from .cider_scorer import CiderScorer

T = TypeVar("T")


class Cider:
    """
    Main Class to compute the CIDEr metric

    """

    def __init__(self, test: Any = None, refs: Any = None, n: int = 4, sigma: float = 6.0) -> None:
        # set cider to sum over 1 to 4-grams
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma

        if test is not None or refs is not None:
            raise NotImplementedError("Cider does not support test and refs arguments")

    def compute_score(self, gts: Dict[T, List[str]], res: Dict[T, List[str]]) -> Tuple[float, Dict[T, float]]:
        """
        Main function to compute CIDEr score
        :param  hypo_for_image (dict) : dictionary with key <image> and value <tokenized hypothesis /
            candidate sentence>
                ref_for_image (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: cider (float) : computed CIDEr score for the corpus
        """

        # assert (list(gts.keys()) == list(res.keys()))
        imgIds = list(res.keys())

        cider_scorer = CiderScorer(n=self._n, sigma=self._sigma)

        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert type(hypo) is list
            assert len(hypo) == 1
            assert type(ref) is list
            assert len(ref) > 0

            cider_scorer += (hypo[0], ref)

        (score, scores) = cider_scorer.compute_score()

        return score, dict(zip(imgIds, scores))

    def method(self) -> str:
        return "CIDEr"


class CiderBase:
    """
    Main Class to compute the CIDEr metric

    """

    def __init__(self, test: Any = None, refs: Any = None, n: int = 4, sigma: float = 6.0) -> None:
        # set cider to sum over 1 to 4-grams
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma

        if test is not None or refs is not None:
            raise NotImplementedError("Cider does not support test and refs arguments")

    def compute_score(self, gts: Dict[T, List[str]], res: Dict[T, List[str]]) -> Tuple[float, List[float]]:
        """
        Main function to compute CIDEr score
        :param  hypo_for_image (dict) : dictionary with key <image> and value <tokenized hypothesis /
             candidate sentence>
                ref_for_image (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: cider (float) : computed CIDEr score for the corpus
        """

        assert gts.keys() == res.keys()
        imgIds = gts.keys()

        cider_scorer = CiderScorer(n=self._n, sigma=self._sigma)

        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert type(hypo) is list
            assert len(hypo) == 1
            assert type(ref) is list
            assert len(ref) > 0

            cider_scorer += (hypo[0], ref)

        (score, scores) = cider_scorer.compute_score()

        return score, scores

    def method(self) -> str:
        return "CIDEr"


class SingleScoreCider:
    def __init__(self) -> None:
        self.cider_scorer = CiderScorer(
            external_df_file=os.path.join(os.path.dirname(__file__), "data", "coco-val.pkl")
        )

    def compute_score(self, hypo: List[str], ref: List[str]) -> float:
        self.cider_scorer.clear()
        self.cider_scorer += (hypo[0], ref)
        score, _ = self.cider_scorer.compute_score()
        return score
