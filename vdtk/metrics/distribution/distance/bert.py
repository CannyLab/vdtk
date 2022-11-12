from functools import lru_cache

from sentence_transformers import SentenceTransformer, util

from . import DistanceFunction

try:
    import torch
except ImportError:
    torch = None  # type: ignore

try:
    import bert_score
except ImportError:
    bert_score = None


class BERTDistance(DistanceFunction):
    def __init__(
        self,
    ) -> None:
        if torch is None:
            raise ImportError("torch is not installed. Please install torch: `pip install torch`")

        self._model = SentenceTransformer("all-MiniLM-L6-v2")

    @lru_cache(None)
    def __call__(self, x: str, y: str) -> float:
        assert torch is not None
        with torch.no_grad():
            embeddings1 = self._model.encode([x], convert_to_tensor=True, show_progress_bar=False)
            embeddings2 = self._model.encode([y], convert_to_tensor=True, show_progress_bar=False)
            assert isinstance(embeddings1, torch.Tensor)
            assert isinstance(embeddings2, torch.Tensor)
            cosine_scores = util.cos_sim(embeddings1, embeddings2).cpu()
            # Return the mean of the trace of cosine scores
            return cosine_scores.reshape(-1).mean().item()


class BERTScoreDistance(DistanceFunction):
    def __init__(
        self,
    ) -> None:
        if bert_score is None:
            raise ImportError("bert_score is not installed. Please install bert_score: `pip install bert_score`")

        self._scorer = bert_score.BERTScorer(lang="en", rescale_with_baseline=True)

    @lru_cache(None)
    def __call__(self, x: str, y: str) -> float:
        assert bert_score is not None
        P, R, F = self._scorer.score([x], [y], batch_size=1)
        return F[0].cpu().item()
