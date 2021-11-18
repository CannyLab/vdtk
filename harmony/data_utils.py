import json
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import spacy

# Initialize the spacy tokenizer
_spacy_nlp = spacy.load("en_core_web_lg")


@dataclass
class Sample:
    _id: str
    split: str
    references: List[str] = field(default_factory=list)
    metadata: Optional[Any] = field(default=None)
    _tokenized_references: Optional[Any] = None

    @property
    def references_tokenized(self) -> Any:
        if self._tokenized_references is None:
            self._tokenized_references = list(_spacy_nlp.pipe(self.references))
        return self._tokenized_references

    @property
    def references_tokenized_text(self) -> List[List[str]]:
        return [[r.text for r in ref] for ref in self.references_tokenized]

    @property
    def references_tokenized_pos(self) -> List[List[Tuple[Any, Any]]]:
        return [[(r.lemma_, r.pos_) for r in ref] for ref in self.references_tokenized]


def load_dataset(dataset_json_path: str) -> List[Sample]:
    """
    Loads the dataset from the json file.
    """
    with open(dataset_json_path) as f:
        dataset = json.load(f)
    return [Sample(**sample) for sample in dataset]
