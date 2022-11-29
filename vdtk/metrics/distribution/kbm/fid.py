from functools import lru_cache
from typing import Any, Dict, Optional, Sequence

import torch
from sentence_transformers import SentenceTransformer

from vdtk.metrics.distribution.scorer import MetricScorer


def _sqrtm(m: torch.Tensor) -> torch.Tensor:
    val, vec = torch.linalg.eig(m)
    return vec @ torch.diag(torch.sqrt(val)) @ vec.inverse()


# calculate frechet inception distance
def calculate_fid(act1: torch.Tensor, act2: torch.Tensor) -> float:
    mu1, sigma1 = torch.mean(act1, dim=0), torch.cov(act1.transpose(0, 1))
    mu2, sigma2 = torch.mean(act2, dim=0), torch.cov(act2.transpose(0, 1))
    ssdiff = torch.sum((mu1 - mu2) ** 2)
    bmm = sigma1 @ sigma2
    covmean2 = _sqrtm(bmm).real
    return (ssdiff + torch.trace(sigma1 + sigma2 - 2.0 * covmean2)).cpu().item()


class FIDMetricScorer(MetricScorer):
    def _initialize_worker_state(self) -> Dict[str, Any]:
        # Initialize the models
        model = SentenceTransformer("all-MiniLM-L6-v2")

        @lru_cache(None)
        def embedding_function(str: str) -> torch.Tensor:
            with torch.no_grad():
                return model.encode([str], convert_to_tensor=True, show_progress_bar=False)[0].cuda()

        return {
            "embedding_function": embedding_function,
        }

    def _score(
        self,
        candidates: Sequence[str],
        references: Sequence[str],
        worker_state: Dict[str, Any],
    ) -> Optional[float]:
        with torch.no_grad():
            # calculate frechet inception distance
            act1 = torch.stack([worker_state["embedding_function"](candidate) for candidate in candidates], dim=0)
            act2 = torch.stack([worker_state["embedding_function"](reference) for reference in references], dim=0)
            fid = calculate_fid(act1, act2)
            return fid
