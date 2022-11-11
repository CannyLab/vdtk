import copy
import itertools
import logging
import random
from dataclasses import dataclass
from typing import (Any, Dict, Generator, List, Mapping, Optional, Sequence,
                    Tuple, TypeVar)

import mpire
import numpy as np
import torch

from .utils.progress import track


def null_track(iterable, *args, **kwargs):
    yield from iterable


def _random_split(data: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Split the data into two random halves"""
    data = list(data)
    random.shuffle(data)
    return data[: len(data) // 2], data[len(data) // 2 :]


def _iterate_partitions(
    data: Sequence[str], partition_a_len: Optional[int] = None, partition_b_len: Optional[int] = None
) -> Generator[Tuple[List[str], List[str]], None, None]:
    for i in range(2 ** (len(data)) - 1):
        binary_string = f"{i:0{len(data)}b}"
        partition_a = [data[i] for i in range(len(data)) if binary_string[i] == "1"]
        partition_b = [data[i] for i in range(len(data)) if binary_string[i] == "0"]
        if len(partition_a) > 0 and len(partition_b) > 0:
            if partition_a_len is not None and len(partition_a) != partition_a_len:
                continue
            if partition_b_len is not None and len(partition_b) != partition_b_len:
                continue
            yield partition_a, partition_b


T = TypeVar("T")


def _upsample(sample: List[T], n: int) -> List[T]:
    upsampled = copy.copy(sample)
    if n > 0:
        while len(upsampled) < n:
            upsampled.append(random.choice(sample))
    return upsampled


@dataclass
class SampleResult:
    p_value: float
    test_statistic: float
    null_hypothesis_samples: Tuple[Optional[float], ...]


class MetricScorer:
    def __init__(
        self,
        num_null_samples: int = 50,
        num_workers: Optional[int] = None,
        log_p: bool = False,
        maintain_worker_state: bool = True,
        quiet: bool = False,
        supersample: bool = False,
    ):
        self._num_null_samples = num_null_samples
        self._num_workers = num_workers or 0
        self._log_p = log_p
        self._worker_state = None
        self._maintain_worker_state = maintain_worker_state
        self._quiet = quiet
        self._supersample = supersample

    def _score(
        self, candidates: Sequence[str], references: Sequence[str], worker_state: Dict[str, Any]
    ) -> Optional[float]:
        """Sub-classes override this function to compute the core value"""
        raise NotImplementedError()

    def _initialize_worker_state(
        self,
    ) -> Dict[str, Any]:
        return {}

    def score(
        self,
        candidates: Sequence[str],
        references: Sequence[str],
        worker_state: Dict[str, Any],
    ) -> Optional[SampleResult]:
        # Handle p-value computations for the data
        sampled_score = self._score(candidates, references, worker_state)
        if sampled_score is None:
            return None

        null_samples = []
        if self._num_null_samples > 0:
            null_refs = list(itertools.chain(candidates, references))
            possible_partitions = list(_iterate_partitions(null_refs, len(candidates), len(references)))
            for _ in range(self._num_null_samples):
                partition_a, partition_b = random.choice(possible_partitions)
                null_samples.append(self._score(partition_a, partition_b, worker_state))

        elif self._num_null_samples == -1:
            # Explicitly compute the null distribution by iterating over all possible partitions
            null_refs = list(itertools.chain(candidates, references))
            for (samples_a, samples_b) in _iterate_partitions(null_refs, len(candidates), len(references)):
                score = self._score(samples_a, samples_b, worker_state)
                if score is not None:
                    null_samples.append(score)

        # Compute a p-value given the number of samples
        p_value = (
            float((np.array(null_samples + [sampled_score]) >= sampled_score - 1e-6).astype(np.float32).mean())
            if null_samples
            else 1.0
        )

        if self._log_p:
            if p_value == 0:
                p_value = 1 / (len(null_samples) + 1e-12)
            p_value = np.log(p_value)
        return SampleResult(p_value, sampled_score, tuple(null_samples))

    def _call_init(self, worker_id: int, worker_state: Dict[str, Any]) -> None:
        # distribute GPUs across workers
        if torch.cuda.is_available():
            torch.cuda.set_device(device=torch.cuda.device(worker_id % torch.cuda.device_count()))

        worker_state["state"] = self._initialize_worker_state()
        worker_state["state"]["worker_id"] = worker_id

    def _call_process(
        self,
        worker_id: int,
        worker_state: Dict[str, Any],
        key: str,
        candidates: Sequence[str],
        references: Sequence[str],
    ):
        try:
            return key, self.score(candidates, references, worker_state["state"])
        except Exception as e:
            if not self._quiet:
                logging.warning(f"Error computing score for {key}: {e}")
            return (key, None)

    def __call__(
        self, candidate_dataset: Mapping[str, Sequence[str]], reference_dataset: Mapping[str, Sequence[str]], _type=0
    ) -> Dict[str, Optional[SampleResult]]:

        if self._supersample:
            # Resample the data until it's large enough
            candidate_dataset = {
                k: _upsample(v, len(reference_dataset.get(k, []))) for k, v in candidate_dataset.items()
            }

        output_scores = {}
        if self._num_workers == 0:
            # Single-threaded operation
            worker_state = (
                self._initialize_worker_state()
                if not self._maintain_worker_state or self._worker_state is None
                else self._worker_state
            )
            if self._maintain_worker_state:
                self._worker_state = worker_state

            for key in (null_track if self._quiet else track)(
                candidate_dataset, description="Computing test-statistics", transient=True
            ):
                if key in reference_dataset:
                    output_scores[key] = self.score(candidate_dataset[key], reference_dataset[key], worker_state)
                else:
                    if not self._quiet:
                        logging.warning(f"{key} not in reference dataset")
            return output_scores

        with mpire.WorkerPool(
            n_jobs=self._num_workers, use_worker_state=True, pass_worker_id=True, start_method="spawn"
        ) as pool:
            args = [
                (k, candidate_dataset[k], reference_dataset[k]) for k in candidate_dataset if k in reference_dataset
            ]
            for sample in (null_track if self._quiet else track)(
                pool.imap_unordered(self._call_process, args, worker_init=self._call_init),
                total=len(args),
                description="Computing test-statistics",
                transient=True,
            ):
                key, result = sample
                output_scores[key] = result

            if not self._quiet:
                logging.debug("Done computing - Worker insights: {}".format(pool.get_insights()))

        return output_scores

    def compute_partition_scores(
        self, dataset: Mapping[str, Sequence[str]], partitions: int = 3
    ) -> Dict[str, List[Optional[SampleResult]]]:

        if self._num_workers == 0:
            worker_state = (
                self._initialize_worker_state()
                if self._worker_state is None and not self._maintain_worker_state
                else self._worker_state
            )
            if self._maintain_worker_state:
                self._worker_state = worker_state
            assert worker_state is not None
            scores: Dict[str, List[Optional[SampleResult]]] = {}
            for i in range(partitions):
                for key, values in (null_track if self._quiet else track)(
                    dataset.items(),
                    total=len(dataset),
                    description="Computing test-statistics for partition {} of {}".format(i + 1, partitions),
                    transient=True,
                ):
                    if key not in scores:
                        scores[key] = []

                    try:
                        # Split samples in half
                        samples_a, samples_b = _random_split(values)
                        scores[key].append(self.score(samples_a, samples_b, worker_state))
                    except Exception as e:
                        if not self._quiet:
                            logging.warning(f"Error computing UK for {key}: {e}")
                        scores[key].append(None)
            return scores

        outputs: Dict[str, List[Optional[SampleResult]]] = {}

        with mpire.WorkerPool(
            n_jobs=self._num_workers, use_worker_state=True, keep_alive=True, pass_worker_id=True, start_method="spawn"
        ) as pool:
            for i in range(partitions):
                if not self._quiet:
                    logging.info("Computing partition {} of {}".format(i + 1, partitions))
                # Compute random partitions of the data
                args = [(k, *_random_split(v)) for k, v in dataset.items()]
                for sample in (null_track if self._quiet else track)(
                    pool.imap_unordered(self._call_process, args, worker_init=self._call_init),
                    total=len(args),
                    description="Computing test-statistics for partition {} of {}".format(i + 1, partitions),
                    transient=True,
                ):
                    key, result = sample
                    if key not in outputs:
                        outputs[key] = []
                    outputs[key].append(result)
            if not self._quiet:
                logging.debug("Done computing partitions - Worker insights: {}".format(pool.get_insights()))
        return outputs
