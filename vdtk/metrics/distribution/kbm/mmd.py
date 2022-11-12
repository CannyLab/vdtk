from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Union

import clip
import embeddings
import torch
from sentence_transformers import SentenceTransformer

from vdtk.metrics.distribution.scorer import MetricScorer


# calculate MMD Distance with a Gaussian kernel
def calculate_mmd(x: torch.Tensor, y: torch.Tensor, sigma: Union[str, float] = "median") -> float:
    # compare kernel MMD paper and code:
    # A. Gretton et al.: A kernel two-sample test, JMLR 13 (2012)
    # http://www.gatsby.ucl.ac.uk/~gretton/mmd/mmd.htm
    # x shape [n, d] y shape [m, d]
    # n_perm number of bootstrap permutations to get p-value, pass none to not get p-value

    dists = torch.pdist(torch.cat([x, y], dim=0))
    c: Optional[Union[torch.Tensor, float]] = None
    if isinstance(sigma, str):
        if sigma == "median":
            c = dists.median() / 2
        elif sigma == "mean":
            c = dists.mean() / 2
        elif sigma == "max":
            c = dists.max() / 2
        elif sigma == "max2":
            c = dists.max()
        else:
            raise ValueError(f"sigma must be 'median', 'mean', 'max', or 'max2', not {sigma}")
    else:
        c = sigma

    assert c is not None, "sigma must be a string or a float"

    n, d = x.shape
    m, d2 = y.shape
    assert d == d2
    xy = torch.cat([x.detach(), y.detach()], dim=0)
    dists = torch.cdist(xy, xy, p=2.0)
    # we are a bit sloppy here as we just keep the diagonal and everything twice
    # note that sigma should be squared in the RBF to match the Gretton et al heuristic
    k = torch.exp((-1 / (2 * c**2)) * dists**2) + torch.eye(n + m).to(dists.device) * 1e-5
    k_x = k[:n, :n]
    k_y = k[n:, n:]
    k_xy = k[:n, n:]
    # The diagonals are always 1 (up to numerical error, this is (3) in Gretton et al.)
    # note that their code uses the biased (and differently scaled mmd)
    mmd = k_x.sum() / (n * (n - 1)) + k_y.sum() / (m * (m - 1)) - 2 * k_xy.sum() / (n * m)
    return mmd.cpu().item()


class MMDBertMetricScorer(MetricScorer):
    def __init__(
        self,
        mmd_sigma: Union[str, float] = "median",
        num_null_samples: int = 50,
        num_workers: Optional[int] = None,
        log_p: bool = False,
        maintain_worker_state: bool = True,
        quiet: bool = False,
        supersample: bool = False,
    ) -> None:
        super().__init__(
            num_null_samples,
            num_workers,
            log_p,
            maintain_worker_state,
            quiet,
            supersample,
        )
        self.mmd_sigma = mmd_sigma

    def _initialize_worker_state(self) -> Dict[str, Any]:
        # Initialize the models
        model = SentenceTransformer("all-MiniLM-L6-v2")

        @lru_cache(None)
        def embedding_function(str: str) -> torch.Tensor:
            with torch.no_grad():
                return model.encode([str], convert_to_tensor=True, show_progress_bar=False)[0]

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
            # calculate MMD
            act1 = torch.stack([worker_state["embedding_function"](candidate) for candidate in candidates], dim=0)
            act2 = torch.stack([worker_state["embedding_function"](reference) for reference in references], dim=0)
            return calculate_mmd(act1, act2, self.mmd_sigma)


class MMDBaseMetricScorer(MetricScorer):
    def __init__(self, mmd_sigma: Union[str, float] = "median", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.mmd_sigma = mmd_sigma

    def _initialize_worker_state(self) -> Dict[str, Any]:
        raise NotImplementedError("MMDBaseMetricScorer._initialize_worker_state")

    def _score(
        self,
        candidates: Sequence[str],
        references: Sequence[str],
        worker_state: Dict[str, Any],
    ) -> Optional[float]:
        with torch.no_grad():
            # calculate MMD
            act1 = torch.stack([worker_state["embedding_function"](candidate) for candidate in candidates], dim=0)
            act2 = torch.stack([worker_state["embedding_function"](reference) for reference in references], dim=0)
            return calculate_mmd(act1, act2, self.mmd_sigma)


class MMDGloveMetricScorer(MMDBaseMetricScorer):
    def _initialize_worker_state(self) -> Dict[str, Any]:
        # Initialize the models
        model = embeddings.GloveEmbedding(default="none")

        @lru_cache(None)
        def embedding_function(str: str) -> torch.Tensor:
            embedded_words = list(filter(lambda x: x is not None, [model.emb(s.lower()) for s in str.split()]))
            if len(embedded_words) == 0:
                return torch.zeros(model.d_emb)
            return torch.stack([torch.FloatTensor(e) for e in embedded_words if e[0] is not None], dim=0).mean(dim=0)

        return {
            "embedding_function": embedding_function,
        }


class MMDFastTextMetricScorer(MMDBaseMetricScorer):
    def _initialize_worker_state(self) -> Dict[str, Any]:
        # Initialize the models
        model = embeddings.FastTextEmbedding(default="none")

        @lru_cache(None)
        def embedding_function(str: str) -> torch.Tensor:
            embedded_words = list(filter(lambda x: x is not None, [model.emb(s.lower()) for s in str.split()]))
            if len(embedded_words) == 0:
                return torch.zeros(model.d_emb)
            return torch.stack([torch.FloatTensor(e) for e in embedded_words if e[0] is not None], dim=0).mean(dim=0)

        return {
            "embedding_function": embedding_function,
        }


class MMDCLIPMetricScorer(MMDBaseMetricScorer):
    def _initialize_worker_state(self) -> Dict[str, Any]:
        # Initialize the models
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)

        @lru_cache(None)
        def embedding_function(str: str) -> torch.Tensor:
            with torch.no_grad():
                text = clip.tokenize([str]).to(device)
                return model.encode_text(text)[0].float()

        return {
            "embedding_function": embedding_function,
        }


class MMDBOWMetricScorer(MMDBaseMetricScorer):
    def __init__(self, vocab: List[str], unk_token: str = "<unk>", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._vocab = {v: i for i, v in enumerate(vocab)}
        self._unk_token = unk_token

    def _initialize_worker_state(self) -> Dict[str, Any]:
        @lru_cache(None)
        def embedding_function(str: str) -> torch.Tensor:
            data = torch.zeros(len(self._vocab) + 1)
            for word in str.lower().split():
                if word in self._vocab:
                    data[self._vocab[word]] += 1
                else:
                    data[self._vocab[self._unk_token]] += 1
            return data

        return {
            "embedding_function": embedding_function,
        }
