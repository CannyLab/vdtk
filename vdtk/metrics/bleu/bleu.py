#!/usr/bin/env python
#
# File Name : bleu.py
#
# Description : Wrapper for BLEU scorer.
#
# Creation Date : 06-01-2015
# Last Modified : Thu 19 Mar 2015 09:13:28 PM PDT
# Authors : Hao Fang <hfang@uw.edu> and Tsung-Yi Lin <tl483@cornell.edu>

from typing import Any, Dict, List, Tuple, TypeVar

from .bleu_scorer import BleuScorer

T = TypeVar("T")


class Bleu:
    def __init__(self, n: int = 4) -> None:
        # default compute Blue score up to 4
        self._n = n
        self._hypo_for_image: Dict[Any, List[str]] = {}
        self.ref_for_image: Dict[Any, List[str]] = {}

    def compute_score(
        self, gts: Dict[T, List[str]], res: Dict[T, List[str]]
    ) -> Tuple[List[float], List[Dict[T, float]]]:
        score, scores = self.compute_score_flat(gts, res)
        return score, [{i: s for i, s in zip(list(res.keys()), scores[n])} for n in range(self._n)]

    def compute_score_flat(
        self, gts: Dict[Any, List[str]], res: Dict[Any, List[str]]
    ) -> Tuple[List[float], List[List[float]]]:

        # assert (set(gts.keys()) == set(res.keys()))
        imgIds = list(res.keys())

        bleu_scorer = BleuScorer(n=self._n)
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert type(hypo) is list
            assert len(hypo) == 1
            assert type(ref) is list
            assert len(ref) >= 1

            bleu_scorer += (hypo[0], ref)

        score, scores = bleu_scorer.compute_score(option="closest", verbose=0)
        return score, scores

    def method(self) -> str:
        return "Bleu"
