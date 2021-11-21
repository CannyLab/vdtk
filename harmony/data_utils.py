import json
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer


class _Evaluators:

    _spacy_model = None
    _sentence_transformer_model = None

    @staticmethod
    def _spacy_nlp():
        if _Evaluators._spacy_model is None:
            _Evaluators._spacy_model = spacy.load("en_core_web_lg")
        return _Evaluators._spacy_model

    @staticmethod
    def _sentence_transformer():
        if _Evaluators._sentence_transformer_model is None:
            _Evaluators._sentence_transformer_model = SentenceTransformer(
                "all-mpnet-base-v2",
            )
        return _Evaluators._sentence_transformer_model


# Lare model initialization
@dataclass
class Sample:
    _id: Optional[str] = None
    split: Optional[str] = None
    references: List[str] = field(default_factory=list)
    metadata: Optional[Any] = field(default=None)

    # Things that are created on the fly
    _tokenized_references: Optional[Any] = None
    _reference_embeddings: Optional[Any] = None

    @property
    def references_tokenized(self) -> Any:
        if self._tokenized_references is None:
            self._tokenized_references = list(_Evaluators._spacy_nlp().pipe(self.references))
        return self._tokenized_references

    @property
    def references_tokenized_text(self) -> List[List[str]]:
        return [[r.text.lower() for r in ref] for ref in self.references_tokenized]

    @property
    def references_tokenized_lemma(self) -> List[List[str]]:
        return [[r.lemma_.lower() for r in ref] for ref in self.references_tokenized]

    @property
    def references_tokenized_pos(self) -> List[List[Tuple[Any, Any]]]:
        return [[(r.lemma_, r.pos_) for r in ref] for ref in self.references_tokenized]

    @property
    def reference_embeddings(self) -> List[np.ndarray]:
        if self._reference_embeddings is None:
            self._reference_embeddings = _Evaluators._sentence_transformer().encode(
                self.references, show_progress_bar=False
            )
        return self._reference_embeddings


def load_dataset(dataset_json_path: str) -> List[Sample]:
    """
    Loads the dataset from the json file.
    """
    with open(dataset_json_path) as f:
        dataset = json.load(f)
    return [Sample(**sample) for sample in dataset]
