import spacy
import nltk
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import itertools
from multiprocessing.pool import Pool
import pickle
import random

# For each sample, compute the fuzzy ratio
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")





class Processor():
    def __init__(self, concepts, cap_matches):
        self._concepts = concepts
        self._cap_matches = cap_matches

    def __call__(self, data):
        i, sample = data
        matching_concepts = set()
        for concept in set(self._concepts):
            for reference in sample['references']:
                if concept in reference:
                    matching_concepts.add(concept)
                    break
        refs = set(sample['references'])
        hypothesis_set = set([a for a in list(itertools.chain.from_iterable([self._cap_matches[k] for k in matching_concepts])) if a not in refs])
        # hypothesis_set = random.sample(hypothesis_set, min(len(hypothesis_set), 100))
        ref_tokens = [s.split() for s in sample['references']]
        hyp_scores = [nltk.translate.bleu_score.sentence_bleu(ref_tokens, hyp.split()) for hyp in hypothesis_set]
        return (i, hyp_scores)

if __name__ == '__main__':

    dataset = './data/mscoco.json'
    data_type = 'coco'

    with open(dataset, 'r') as jf:
        reference_data = json.load(jf)

    # Load the concept set for Kinetics
    if data_type == 'k600':
        concepts = []
        with open('./data/kinetics_600_labels.csv', 'r') as cf:
            for line in cf:
                concepts.append(line.strip())
    elif data_type == 'k400':
        concepts = []
        with open('./data/kinetics_400_labels.csv', 'r') as cf:
            for line in cf:
                concepts.append(line.strip())
    elif data_type == 'coco':
        concepts = []
        with open('./data/coco_labels.csv', 'r') as cf:
            for line in cf:
                concepts.append(line.strip())
    elif data_type == 'places':
        concepts = []
        with open('./data/places_labels.txt', 'r') as cf:
            for line in cf:
                concept, _ = line.split()
                concepts.append(concept.split('/')[2].replace('_', ' '))
    elif data_type == 'imagenet':
        with open('./data/imagenet_labels.json', 'r') as jf:
            concepts = list(itertools.chain.from_iterable([[s.strip().lower() for s in a.split(',')] for a in list(json.load(jf).values())]))
    else:
        raise ValueError('Unknown data type')

    matched = 0
    matched_concepts = defaultdict(set)
    for sample in tqdm(reference_data):
        sample_matched = False
        for concept in set(concepts):
            for reference in sample['references']:
                if concept in reference:
                    sample_matched = True
                    matched_concepts[concept].add(reference)
        if sample_matched:
            matched += 1

    print('Match', matched / len(reference_data))

    with Pool(28) as pool:
        output_data = []
        for hyp_scores in tqdm(pool.imap_unordered(Processor(concepts, matched_concepts), enumerate(reference_data)), total=len(reference_data)):
            output_data.append(hyp_scores)

    # Sort the hyp scores
    sample_scores = [o[1] for o in sorted(output_data, key=lambda x: x[0])]

    with open('concept_results.pkl', 'wb') as pfile:
        pickle.dump(sample_scores, pfile)

    # plt.hist([np.max(a) for a in sample_scores if len(a) > 0 and np.max(a) < 1], bins=80)
    print('Value', np.mean([np.max(a) for a in sample_scores if len(a) > 0 and np.max(a) < 1]))
