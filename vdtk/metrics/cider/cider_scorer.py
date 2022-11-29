#!/usr/bin/env python
# Tsung-Yi Lin <tl483@cornell.edu>
# Ramakrishna Vedantam <vrama91@vt.edu>

import copy
import math
import pickle
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from vdtk.metrics.bleu.bleu_scorer import precook


def cook_refs(refs: List[str], n: int = 4) -> List[Dict[Tuple[str, ...], int]]:
    return [precook(ref, n)[1] for ref in refs]


def cook_test(test: str, n: int = 4) -> Dict[Tuple[str, ...], int]:
    return precook(test, n, True)[1]


class CiderScorer(object):
    """CIDEr scorer."""

    def copy(self) -> "CiderScorer":
        """copy the refs."""
        new = CiderScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        return new

    def __init__(
        self,
        test: Optional[str] = None,
        refs: Optional[List[str]] = None,
        n: int = 4,
        sigma: float = 6.0,
        external_df_file: Optional[str] = None,
        external_df_corpus_reflen: Optional[int] = None,
    ) -> None:
        """singular instance"""
        self.n = n
        self.sigma = sigma
        self.crefs: List[List[Dict[Tuple[str, ...], int]]] = []
        self.ctest: List[Optional[Dict[Tuple[str, ...], int]]] = []

        self._external_df_file = external_df_file
        if self._external_df_file is None:
            self.document_frequency = defaultdict(float)
            self.ref_len = None
        else:
            with open(self._external_df_file, "rb") as f:
                u = pickle._Unpickler(f)
                u.encoding = "latin1"  # type: ignore
                pkl_file = u.load()
                self.ref_len = external_df_corpus_reflen  # np.log(float(40504))
                self.document_frequency = pkl_file

        self.cook_append(test, refs)

    def cook_append(self, test: Optional[str], refs: Optional[List[str]]) -> None:
        """called by constructor and __iadd__ to avoid creating new instances."""
        if refs is not None:
            self.crefs.append(cook_refs(refs))
            if test is not None:
                self.ctest.append(cook_test(test))
            else:
                self.ctest.append(None)

    def clear(self) -> None:
        self.crefs = []
        self.ctest = []

    def size(self) -> int:
        assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
        return len(self.crefs)

    def __iadd__(self, other: Union[Tuple[str, List[str]], "CiderScorer"]) -> "CiderScorer":
        """add an instance (e.g., from another sentence)."""
        if isinstance(other, CiderScorer):
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)
        else:
            self.cook_append(other[0], other[1])

        return self

    def compute_doc_freq(self) -> None:
        """
        Compute term frequency for reference data.
        This will be used to compute idf (inverse document frequency later)
        The term frequency is stored in the object
        :return: None
        """
        for refs in self.crefs:
            # refs, k ref captions of one image
            for ngram in set([ngram for ref in refs for (ngram, count) in ref.items()]):
                self.document_frequency[ngram] += 1
            # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)

    def compute_cider(self) -> List[float]:
        def counts2vec(cnts: Dict[Tuple[str, ...], int]) -> Tuple[List[Dict[Tuple[str, ...], float]], List[float], int]:
            """
            Function maps counts of ngram to vector of tfidf weights.
            The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
            The n-th entry of array denotes length of n-grams.
            :param cnts:
            :return: vec (array of dict), norm (array of float), length (int)
            """
            vec: List[Dict[Tuple[str, ...], float]] = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram, term_freq) in cnts.items():
                # give word count 1 if it doesn't appear in reference corpus
                df = np.log(max(1.0, self.document_frequency[ngram]))
                # ngram index
                n = len(ngram) - 1
                # tf (term_freq) * idf (precomputed idf) for n-grams
                vec[n][ngram] = float(term_freq) * (self.ref_len - df)
                # compute norm for the vector.  the norm will be used for computing similarity
                norm[n] += pow(vec[n][ngram], 2)

                if n == 1:
                    length += term_freq
            norm = [float(np.sqrt(n)) for n in norm]

            return vec, norm, length

        def sim(
            vec_hyp: List[Dict[Tuple[str, ...], float]],
            vec_ref: List[Dict[Tuple[str, ...], float]],
            norm_hyp: List[float],
            norm_ref: List[float],
            length_hyp: int,
            length_ref: int,
        ) -> np.ndarray:
            """
            Compute the cosine similarity of two vectors.
            :param vec_hyp: array of dictionary for vector corresponding to hypothesis
            :param vec_ref: array of dictionary for vector corresponding to reference
            :param norm_hyp: array of float for vector corresponding to hypothesis
            :param norm_ref: array of float for vector corresponding to reference
            :param length_hyp: int containing length of hypothesis
            :param length_ref: int containing length of reference
            :return: array of score for each n-grams cosine similarity
            """
            delta = float(length_hyp - length_ref)
            # measure consine similarity
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                # ngram
                for (ngram, count) in vec_hyp[n].items():
                    # vrama91 : added clipping
                    val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]

                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= norm_hyp[n] * norm_ref[n]

                assert not math.isnan(val[n])
                # vrama91: added a length based gaussian penalty
                val[n] *= np.e ** (-(delta**2) / (2 * self.sigma**2))
            return val

        # compute log reference length
        if self._external_df_file is None:
            self.ref_len = np.log(float(len(self.crefs)))

        scores: List[float] = []
        for test, refs in zip(self.ctest, self.crefs):
            if test is None:
                continue
            # compute vector for test captions
            vec, norm, length = counts2vec(test)
            # compute vector for ref captions
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            # change by vrama91 - mean of ngram scores, instead of sum
            score_avg = np.mean(score)
            # divide by number of references
            score_avg /= len(refs)
            # multiply score by 10
            score_avg *= 10.0
            # append score of an image to the score list
            scores.append(float(score_avg))
        return scores

    def compute_score(self, option: Any = None, verbose: Any = 0) -> Tuple[float, List[float]]:
        # compute idf
        if self._external_df_file is None:
            self.compute_doc_freq()
        # assert to check document frequency
        try:
            if len(self.ctest) < max(self.document_frequency.values()):
                return float(np.nan), np.array([np.nan for _ in self.ctest]).tolist()
        except ValueError:
            return float(np.nan), np.array([np.nan for _ in self.ctest]).tolist()
        # compute cider score
        score = self.compute_cider()
        # debug
        # print score
        return float(np.mean(np.array(score))), np.array(score).tolist()
