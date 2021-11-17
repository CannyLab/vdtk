

import nltk
import numpy as np
import matplotlib.pyplot as plt

import tqdm
import pickle
import spacy

from collections import Counter
import textdistance
import itertools
import json
from fuzzywuzzy import fuzz
import nlgeval
import random
import multiprocessing
from absl import logging

from typing import List

from harmony.lm import NGramLM


def _normalize(sentences: List[str]) -> List[str]:
    return [sentence.lower().strip() for sentence in sentences]

_ng = None
def _sseval_mp_thread(data):
    gt_limit, samples = data

    global _ng
    if not _ng:
        _ng = nlgeval.NLGEval(no_skipthoughts=True, no_glove=True)
    s_idx = (random.randint(0, len(t)-1) for t in samples)
    ccx = [(t[idx], random.sample(list(set(t) - {t[idx]}), min(len(list(set(t) - {t[idx]})), gt_limit)) if gt_limit > 0 else list(set(t) - {t[idx]})) for t, idx in zip(samples, s_idx)]
    hyps = [a[0] for a in ccx]
    refs = zip(*[a[1] for a in ccx])
    return _ng.compute_metrics(refs, hyps)



class Evaluator():

    def __init__(self, json_file: str):
        # Load the JSON file corresponding to the dataset
        with open(json_file, 'r') as jf:
            self._dataset_data = json.load(jf)

        # Compute spacy tokens
        logging.info(f'Building tokenized samples from {len(self._dataset_data)} samples')
        self._nlp = spacy.load("en_core_web_sm")

        for sample in tqdm.tqdm(self._dataset_data):
            sample['references_tokenized'] = list(self._nlp.pipe(sample['references']))

        # Split the samples into train, validation and test
        self.train_samples = [d for d in self._dataset_data if d['split'] == 'train']
        self.validation_samples = [d for d in self._dataset_data if d['split'] == 'validate']
        self.test_samples = [d for d in self._dataset_data if d['split'] == 'test']

        self._train_vocab = None
        self._validation_vocab = None
        self._test_vocab = None

        self._ngram_models = {
            'train': {},
            'validation': {},
            'test': {}
        }

    @property
    def train_vocab(self):
        if self._train_vocab is None:
            self._train_vocab = Counter(
                itertools.chain.from_iterable([itertools.chain.from_iterable([[t.text for t in s] for s in sample['references_tokenized']]) for sample in self.train_samples])
            )
        return self._train_vocab

    @property
    def validation_vocab(self):
        if self._validation_vocab is None:
            self._validation_vocab = Counter(
                itertools.chain.from_iterable([itertools.chain.from_iterable([[t.text for t in s] for s in sample['references_tokenized']]) for sample in self.validation_samples])
            )
        return self._validation_vocab

    @property
    def test_vocab(self):
        if self._test_vocab is None:
            self._test_vocab = Counter(
                itertools.chain.from_iterable([itertools.chain.from_iterable([[t.text for t in s] for s in sample['references_tokenized']]) for sample in self.test_samples])
            )
        return self._test_vocab


    ######################################################################################
    #  Vocab Uniqueness                                                                 #
    ######################################################################################

    @property
    def train_vocab_uniqueness(self):
        """Return the within-sample, and between sample uniqueness of the training set"""
        vis = []
        vus = []
        for sample in self.train_samples:
            sample_counts = Counter(itertools.chain.from_iterable([[t.text for t in s] for s in sample['references_tokenized']]))
            vis.append(len([i for i in sample_counts.keys() if sample_counts[i] == 1]) / (1e-9 + sum(sample_counts.values())))
            vus.append(len([i for i in self.train_vocab.keys() if self.train_vocab[i] == 1]) / (1e-9 + sum(self.train_vocab.values())))
        return np.mean(vus), np.mean(vis)

    @property
    def validation_vocab_uniqueness(self):
        """Return the within-sample, and between sample uniqueness of the validation set"""
        vis = []
        vus = []
        for sample in self.validation_samples:
            sample_counts = Counter(itertools.chain.from_iterable([[t.text for t in s] for s in sample['references_tokenized']]))
            vis.append(len([i for i in sample_counts.keys() if sample_counts[i] == 1]) / (1e-9 + sum(sample_counts.values())))
            vus.append(len([i for i in self.validation_vocab.keys() if self.validation_vocab[i] == 1]) / (1e-9 + sum(self.validation_vocab.values())))
        return np.mean(vus), np.mean(vis)

    @property
    def test_vocab_uniqueness(self):
        """Return the within-sample, and between sample uniqueness of the test set"""
        vis = []
        vus = []
        for sample in self.test_samples:
            sample_counts = Counter(itertools.chain.from_iterable([[t.text for t in s] for s in sample['references_tokenized']]))
            vis.append(len([i for i in sample_counts.keys() if sample_counts[i] == 1]) / (1e-9 + sum(sample_counts.values())))
            vus.append(len([i for i in self.test_vocab.keys() if self.test_vocab[i] == 1]) / (1e-9 + sum(self.test_vocab.values())))
        return np.mean(vus), np.mean(vis)

    ######################################################################################
    #  Vocab Usage                                                                       #
    ######################################################################################

    @property
    def validation_vocab_usage(self):
        # Return how much of the training vocab is used in the validation vocab
        return len([a for a in self.train_vocab if a in self.validation_vocab]) / len(self.train_vocab)

    @property
    def test_vocab_usage(self):
        # Return how much of the training vocab is used in the test vocab
        return len([a for a in self.train_vocab if a in self.test_vocab]) / len(self.train_vocab)

    ######################################################################################
    #  POS Counts                                                                        #
    ######################################################################################

    @property
    def train_pos_counts(self):
        # Return the POS counts for the training set
        return Counter(
                itertools.chain.from_iterable([itertools.chain.from_iterable([[t.pos_ for t in s] for s in sample['references_tokenized']]) for sample in self.train_samples])
            )

    @property
    def validation_pos_counts(self):
        # Return the POS counts for the validation set
        return Counter(
                itertools.chain.from_iterable([itertools.chain.from_iterable([[t.pos_ for t in s] for s in sample['references_tokenized']]) for sample in self.validation_samples])
            )

    @property
    def test_pos_counts(self):
        # Return the POS counts for the test set
        return Counter(
                itertools.chain.from_iterable([itertools.chain.from_iterable([[t.pos_ for t in s] for s in sample['references_tokenized']]) for sample in self.test_samples])
            )

    @property
    def train_noun_counts(self):
        # Return the noun counts for the training set
        return Counter(
                itertools.chain.from_iterable([itertools.chain.from_iterable([[t.lemma_ for t in s if t.pos_ in ('NOUN', 'PROPN')] for s in sample['references_tokenized']]) for sample in self.train_samples])
            )

    @property
    def validation_noun_counts(self):
        # Return the noun counts for the validation set
        return Counter(
                itertools.chain.from_iterable([itertools.chain.from_iterable([[t.lemma_ for t in s if t.pos_ in ('NOUN', 'PROPN')] for s in sample['references_tokenized']]) for sample in self.validation_samples])
            )

    @property
    def test_noun_counts(self):
        # Return the noun counts for the test set
        return Counter(
                itertools.chain.from_iterable([itertools.chain.from_iterable([[t.lemma_ for t in s if t.pos_ in ('NOUN', 'PROPN')] for s in sample['references_tokenized']]) for sample in self.test_samples])
            )

    @property
    def train_verb_counts(self):
        # Return the verb counts for the training set
        return Counter(
                itertools.chain.from_iterable([itertools.chain.from_iterable([[t.lemma_ for t in s if t.pos_ == 'VERB'] for s in sample['references_tokenized']]) for sample in self.train_samples])
            )

    @property
    def validation_verb_counts(self):
        # Return the verb counts for the validation set
        return Counter(
                itertools.chain.from_iterable([itertools.chain.from_iterable([[t.lemma_ for t in s if t.pos_ == 'VERB'] for s in sample['references_tokenized']]) for sample in self.validation_samples])
            )

    @property
    def test_verb_counts(self):
        # Return the verb counts for the test set
        return Counter(
                itertools.chain.from_iterable([itertools.chain.from_iterable([[t.lemma_ for t in s if t.pos_ == 'VERB'] for s in sample['references_tokenized']]) for sample in self.test_samples])
            )

    ######################################################################################
    #  Noun/Verb Uniqueness                                                              #
    ######################################################################################

    @property
    def train_noun_uniqueness(self):
        # Return the uniqueness of nouns within the sample, and within the dataset
        base_noun_counts = self.train_noun_counts
        vis = []
        vus = []
        for sample in self.train_samples:
            sample_noun_counts = Counter(itertools.chain.from_iterable([[t.lemma_ for t in s if t.pos_ in ('NOUN', 'PROPN')] for s in sample['references_tokenized']]))
            vis.append(len([i for i in sample_noun_counts.keys() if sample_noun_counts[i] == 1]) / (1e-9 + sum(sample_noun_counts.values())))
            vus.append(len([i for i in sample_noun_counts.keys() if base_noun_counts[i] == 1]) / (1e-9 + sum(sample_noun_counts.values())))

        return np.mean(vis), np.mean(vus)

    @property
    def train_verb_uniqueness(self):
        # Return the uniqueness of verbs within the sample, and within the dataset
        base_verb_counts = self.train_verb_counts
        vis = []
        vus = []
        for sample in self.train_samples:
            sample_verb_counts = Counter(itertools.chain.from_iterable([[t.lemma_ for t in s if t.pos_ == 'VERB'] for s in sample['references_tokenized']]))
            vis.append(len([i for i in sample_verb_counts.keys() if sample_verb_counts[i] == 1]) / (1e-9 + sum(sample_verb_counts.values())))
            vus.append(len([i for i in sample_verb_counts.keys() if base_verb_counts[i] == 1]) / (1e-9 + sum(sample_verb_counts.values())))

        return np.mean(vis), np.mean(vus)

    @property
    def validation_noun_uniqueness(self):
        # Return the uniqueness of nouns within the sample, and within the dataset
        base_noun_counts = self.validation_noun_counts
        vis = []
        vus = []
        for sample in self.validation_samples:
            sample_noun_counts = Counter(itertools.chain.from_iterable([[t.lemma_ for t in s if t.pos_ in ('NOUN', 'PROPN')] for s in sample['references_tokenized']]))
            vis.append(len([i for i in sample_noun_counts.keys() if sample_noun_counts[i] == 1]) / (1e-9 + sum(sample_noun_counts.values())))
            vus.append(len([i for i in sample_noun_counts.keys() if base_noun_counts[i] == 1]) / (1e-9 + sum(sample_noun_counts.values())))

        return np.mean(vis), np.mean(vus)

    @property
    def validation_verb_uniqueness(self):
        # Return the uniqueness of verbs within the sample, and within the dataset
        base_verb_counts = self.validation_verb_counts
        vis = []
        vus = []
        for sample in self.validation_samples:
            sample_verb_counts = Counter(itertools.chain.from_iterable([[t.lemma_ for t in s if t.pos_ == 'VERB'] for s in sample['references_tokenized']]))
            vis.append(len([i for i in sample_verb_counts.keys() if sample_verb_counts[i] == 1]) / (1e-9 + sum(sample_verb_counts.values())))
            vus.append(len([i for i in sample_verb_counts.keys() if base_verb_counts[i] == 1]) / (1e-9 + sum(sample_verb_counts.values())))

        return np.mean(vis), np.mean(vus)

    @property
    def test_noun_uniqueness(self):
        # Return the uniqueness of nouns within the sample, and within the dataset
        base_noun_counts = self.test_noun_counts
        vis = []
        vus = []
        for sample in self.test_samples:
            sample_noun_counts = Counter(itertools.chain.from_iterable([[t.lemma_ for t in s if t.pos_ in ('NOUN', 'PROPN')] for s in sample['references_tokenized']]))
            vis.append(len([i for i in sample_noun_counts.keys() if sample_noun_counts[i] == 1]) / (1e-9 + sum(sample_noun_counts.values())))
            vus.append(len([i for i in sample_noun_counts.keys() if base_noun_counts[i] == 1]) / (1e-9 + sum(sample_noun_counts.values())))

        return np.mean(vis), np.mean(vus)

    @property
    def test_verb_uniqueness(self):
        # Return the uniqueness of verbs within the sample, and within the dataset
        base_verb_counts = self.test_verb_counts
        vis = []
        vus = []
        for sample in self.test_samples:
            sample_verb_counts = Counter(itertools.chain.from_iterable([[t.lemma_ for t in s if t.pos_ == 'VERB'] for s in sample['references_tokenized']]))
            vis.append(len([i for i in sample_verb_counts.keys() if sample_verb_counts[i] == 1]) / (1e-9 + sum(sample_verb_counts.values())))
            vus.append(len([i for i in sample_verb_counts.keys() if base_verb_counts[i] == 1]) / (1e-9 + sum(sample_verb_counts.values())))

        return np.mean(vis), np.mean(vus)

    ######################################################################################
    #  N-Gram Models                                                                     #
    ######################################################################################


    def _build_ngram_model(self, samples: list, n: int):
        return NGramLM(itertools.chain.from_iterable([ [[t.text for t in s] for s in sample['references_tokenized']] for sample in samples]), n)

    def train_ngram_model(self, n: int):
        if self._ngram_models['train'].get(n, None) is None:
            self._ngram_models['train'][n] = self._build_ngram_model(self.train_samples, n)
        return self._ngram_models['train'][n]

    def validation_ngram_model(self, n: int):
        if self._ngram_models['validation'].get(n, None) is None:
            self._ngram_models['validation'][n] = self._build_ngram_model(self.validation_samples, n)
        return self._ngram_models['validation'][n]

    def test_ngram_model(self, n: int):
        if self._ngram_models['test'].get(n, None) is None:
            self._ngram_models['test'][n] = self._build_ngram_model(self.test_samples, n)
        return self._ngram_models['test'][n]

    def train_ngram_count(self, n: int):
        return self.train_ngram_model(n).count

    def validation_ngram_count(self, n: int):
        return self.validation_ngram_model(n).count

    def test_ngram_count(self, n: int):
        return self.test_ngram_model(n).count

    ######################################################################################
    #  N-Gram EVS                                                                        #
    ######################################################################################

    def train_evs(self, n: int):
        # Determine how many n-grams act like 1-grams in the training set
        train_model = self.train_ngram_model(n)
        return len([i for i in train_model.model.keys() if len(train_model.model[i]) <= 1]) / len(train_model.model)

    def validation_evs(self, n: int):
        # Determine how many n-grams act like 1-grams in the validation set
        validation_model = self.validation_ngram_model(n)
        return len([i for i in validation_model.model.keys() if len(validation_model.model[i]) <= 1]) / len(validation_model.model)

    def test_evs(self, n: int):
        # Determine how many n-grams act like 1-grams in the test set
        test_model = self.test_ngram_model(n)
        return len([i for i in test_model.model.keys() if len(test_model.model[i]) <= 1]) / len(test_model.model)

    ######################################################################################
    #  N-Gram Perplexity                                                                 #
    ######################################################################################

    def train_ngram_ll(self, n: int):
        train_model = self.train_ngram_model(n)
        return np.mean(list(itertools.chain.from_iterable([[train_model.log_likelihood([t.text for t in s]) for s in sample['references_tokenized']] for sample in self.train_samples])))

    def validation_ngram_ll(self, n: int):
        train_model = self.train_ngram_model(n)
        return np.mean(list(itertools.chain.from_iterable([[train_model.log_likelihood([t.text for t in s]) for s in sample['references_tokenized']] for sample in self.validation_samples])))

    def test_ngram_ll(self, n: int):
        train_model = self.train_ngram_model(n)
        return np.mean(list(itertools.chain.from_iterable([[train_model.log_likelihood([t.text for t in s]) for s in sample['references_tokenized']] for sample in self.test_samples])))

    ######################################################################################
    #  Caption Lengths                                                                   #
    ######################################################################################

    @property
    def train_caption_lengths(self):
        return [[len(s) for s in sample['references_tokenized']] for sample in self.train_samples]

    @property
    def validation_caption_lengths(self):
        return [[len(s) for s in sample['references_tokenized']] for sample in self.validation_samples]

    @property
    def test_caption_lengths(self):
        return [[len(s) for s in sample['references_tokenized']] for sample in self.test_samples]

    ######################################################################################
    #  Verbs / Nouns per caption                                                         #
    ######################################################################################

    @property
    def train_verbs_per_caption(self):
        return [[len([t for t in s if t.pos_ == 'VERB']) for s in sample['references_tokenized']] for sample in self.train_samples]

    @property
    def validation_verbs_per_caption(self):
        return [[len([t for t in s if t.pos_ == 'VERB']) for s in sample['references_tokenized']] for sample in self.validation_samples]

    @property
    def test_verbs_per_caption(self):
        return [[len([t for t in s if t.pos_ == 'VERB']) for s in sample['references_tokenized']] for sample in self.test_samples]

    @property
    def train_nouns_per_caption(self):
        return [[len([t for t in s if t.pos_ in ('NOUN', 'PROPN')]) for s in sample['references_tokenized']] for sample in self.train_samples]

    @property
    def validation_nouns_per_caption(self):
        return [[len([t for t in s if t.pos_ in ('NOUN', 'PROPN')]) for s in sample['references_tokenized']] for sample in self.validation_samples]

    @property
    def test_nouns_per_caption(self):
        return [[len([t for t in s if t.pos_ in ('NOUN', 'PROPN')]) for s in sample['references_tokenized']] for sample in self.test_samples]

    ######################################################################################
    #  Caption novelty / Uniqueness                                                      #
    ######################################################################################

    @property
    def within_sample_train_caption_novelty(self):
        return [len(set(_normalize(sample['references']))) / len(_normalize(sample['references'])) for sample in self.train_samples]

    @property
    def  within_sample_validation_caption_novelty(self):
        return [len(set(_normalize(sample['references']))) / len(_normalize(sample['references'])) for sample in self.validation_samples]

    @property
    def  within_sample_test_caption_novelty(self):
        return [len(set(_normalize(sample['references']))) / len(_normalize(sample['references'])) for sample in self.test_samples]

    @property
    def between_sample_train_caption_novelty(self):
        all_references = set(itertools.chain.from_iterable([_normalize(sample['references']) for sample in self.train_samples]))
        return len(all_references) / len(list(itertools.chain.from_iterable([_normalize(sample['references']) for sample in self.train_samples])))

    @property
    def between_sample_validation_caption_novelty(self):
        all_references = set(itertools.chain.from_iterable([_normalize(sample['references']) for sample in self.validation_samples]))
        return len(all_references) / len(list(itertools.chain.from_iterable([_normalize(sample['references']) for sample in self.validation_samples])))

    @property
    def between_sample_test_caption_novelty(self):
        all_references = set(itertools.chain.from_iterable([_normalize(sample['references']) for sample in self.test_samples]))
        return len(all_references) / len(list(itertools.chain.from_iterable([_normalize(sample['references']) for sample in self.test_samples])))

    @property
    def train_unique_caption_counts(self):
        return [len(set(_normalize(sample['references']))) for sample in self.test_samples]

    @property
    def validation_unique_caption_counts(self):
        return [len(set(_normalize(sample['references']))) for sample in self.validation_samples]

    @property
    def test_unique_caption_counts(self):
        return [len(set(_normalize(sample['references']))) for sample in self.test_samples]



    ######################################################################################
    #  Within-Sample Scoring                                                             #
    ######################################################################################


    def train_within_sample_score(self, samples: int = 750, processes: int = 8, gt_limit: int = -1):
        tokenized_samples = (gt_limit, [[' '.join([t.text for t in s]) for s in sample['references_tokenized']] for sample in self.train_samples])
        scores = []
        with multiprocessing.Pool(processes=processes) as pool:
            for score in tqdm.tqdm(pool.imap_unordered(_sseval_mp_thread, itertools.repeat(tokenized_samples, samples)), total=samples):
                scores.append(score)
        return scores

    def validation_within_sample_score(self, samples: int = 750, processes: int = 8, gt_limit: int = -1):
        tokenized_samples = (gt_limit, [[' '.join([t.text for t in s]) for s in sample['references_tokenized']] for sample in self.validation_samples])
        scores = []
        with multiprocessing.Pool(processes=processes) as pool:
            for score in tqdm.tqdm(pool.imap_unordered(_sseval_mp_thread, itertools.repeat(tokenized_samples, samples)), total=samples):
                scores.append(score)
        return scores

    def test_within_sample_score(self, samples: int = 750, processes: int = 8, gt_limit: int = -1):
        tokenized_samples = (gt_limit, [[' '.join([t.text for t in s]) for s in sample['references_tokenized']] for sample in self.test_samples])
        scores = []
        with multiprocessing.Pool(processes=processes) as pool:
            for score in tqdm.tqdm(pool.imap_unordered(_sseval_mp_thread, itertools.repeat(tokenized_samples, samples)), total=samples):
                scores.append(score)
        return scores

    def train_within_sample_score_masked(self, samples: int = 750, processes: int = 8, gt_limit: int = -1):
        class uq:
            value = 0
            @classmethod
            def increment(cls):
                cls.value += 1
                return cls.value

        tokenized_samples = (gt_limit, [[' '.join([t.text if t.pos_ not in ('NOUN', 'PROPN', 'VERB') else f'MASK{uq.increment()}' for t in s]) for s in sample['references_tokenized']] for sample in self.train_samples])
        scores = []
        with multiprocessing.Pool(processes=processes) as pool:
            for score in tqdm.tqdm(pool.imap_unordered(_sseval_mp_thread, itertools.repeat(tokenized_samples, samples)), total=samples):
                scores.append(score)
        return scores

    def validation_within_sample_score_masked(self, samples: int = 750, processes: int = 8, gt_limit: int = -1):
        class uq:
            value = 0
            @classmethod
            def increment(cls):
                cls.value += 1
                return cls.value

        tokenized_samples = (gt_limit, [[' '.join([t.text if t.pos_ not in ('NOUN', 'PROPN', 'VERB') else f'MASK{uq.increment()}' for t in s]) for s in sample['references_tokenized']] for sample in self.validation_samples])
        scores = []
        with multiprocessing.Pool(processes=processes) as pool:
            for score in tqdm.tqdm(pool.imap_unordered(_sseval_mp_thread, itertools.repeat(tokenized_samples, samples)), total=samples):
                scores.append(score)
        return scores

    def test_within_sample_score_masked(self, samples: int = 750, processes: int = 8, gt_limit: int = -1):

        class uq:
            value = 0
            @classmethod
            def increment(cls):
                cls.value += 1
                return cls.value

        tokenized_samples = (gt_limit, [[' '.join([t.text if t.pos_ not in ('NOUN', 'PROPN', 'VERB') else f'MASK{uq.increment()}' for t in s]) for s in sample['references_tokenized']] for sample in self.test_samples])
        scores = []
        with multiprocessing.Pool(processes=processes) as pool:
            for score in tqdm.tqdm(pool.imap_unordered(_sseval_mp_thread, itertools.repeat(tokenized_samples, samples)), total=samples):
                scores.append(score)
        return scores

    ######################################################################################
    #  Category Scoring                                                           #
    ######################################################################################

    def _compute_categories(self, samples):
        pass

    def train_within_category_score(self, metric: str):
        pass

    def save(self, filepath: str):
        logging.info(f'Saving evaluator to {filepath}')
        # Write this object as a pickle to the given filepath
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str):
        logging.info(f'Loading Evaluator from {filepath}')
        # Load a pickle from the given filepath
        with open(filepath, 'rb') as f:
            return pickle.load(f)
