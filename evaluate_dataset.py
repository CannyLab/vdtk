
import os
import pathlib
import pickle
from absl import app, flags, logging
import time

from harmony.evaluator import Evaluator


flags.DEFINE_string('json', None, 'Path to the json file of the dataset.')
flags.DEFINE_string('cache_dir', './cache', 'Path to the cache directory.')
flags.DEFINE_bool('no_cache_dir', False, 'Do not use the cache directory.')
FLAGS = flags.FLAGS

def main(*unused_argv):
    # Load the evaluator from the cache if necessary
    cache_dir = pathlib.Path(FLAGS.cache_dir)
    cache_dir.mkdir(exist_ok=True)
    if not FLAGS.no_cache_dir and (cache_dir / (os.path.basename(FLAGS.json) + '.pkl')).exists():
        evaluator = Evaluator.load(cache_dir / (os.path.basename(FLAGS.json) + '.pkl'))
    else:
        evaluator = Evaluator(FLAGS.json)

    start_time = time.time()
    outputs = {
        'train_vocab': evaluator.train_vocab,
        'validation_vocab': evaluator.validation_vocab,
        'train_vocab_uniqueness': evaluator.train_vocab_uniqueness,
        'validation_vocab_uniqueness': evaluator.validation_vocab_uniqueness,
        'validation_vocab_usage': evaluator.validation_vocab_usage,
        'train_pos_counts': evaluator.train_pos_counts,
        'validation_pos_counts': evaluator.validation_pos_counts,
        'train_noun_counts': evaluator.train_noun_counts,
        'validation_noun_counts': evaluator.validation_noun_counts,
        'train_verb_counts': evaluator.train_verb_counts,
        'validation_verb_counts': evaluator.validation_verb_counts,
        'train_noun_uniqueness': evaluator.train_noun_uniqueness,
        'validation_noun_uniqueness': evaluator.validation_noun_uniqueness,
        'train_verb_uniqueness': evaluator.train_verb_uniqueness,
        'validation_verb_uniqueness': evaluator.validation_verb_uniqueness,
        'train_evs_2': evaluator.train_evs(2),
        'validation_evs_2': evaluator.validation_evs(2),
        'train_evs_3': evaluator.train_evs(3),
        'validation_evs_3': evaluator.validation_evs(3),
        'train_evs_4': evaluator.train_evs(4),
        'validation_evs_4': evaluator.validation_evs(4),
        'train_ngram_ll_2': evaluator.train_ngram_ll(2),
        'validation_ngram_ll_2': evaluator.validation_ngram_ll(2),
        'train_ngram_ll_3': evaluator.train_ngram_ll(3),
        'validation_ngram_ll_3': evaluator.validation_ngram_ll(3),
        'train_ngram_ll_4': evaluator.train_ngram_ll(4),
        'validation_ngram_ll_4': evaluator.validation_ngram_ll(4),
        'train_ngram_count_2': evaluator.train_ngram_count(2),
        'validation_ngram_count_2': evaluator.validation_ngram_count(2),
        'train_ngram_count_3': evaluator.train_ngram_count(3),
        'validation_ngram_count_3': evaluator.validation_ngram_count(3),
        'train_ngram_count_4': evaluator.train_ngram_count(4),
        'validation_ngram_count_4': evaluator.validation_ngram_count(4),
        'train_caption_lengths': evaluator.train_caption_lengths,
        'validation_caption_lengths': evaluator.validation_caption_lengths,
        'train_verbs_per_caption': evaluator.train_verbs_per_caption,
        'validation_verbs_per_caption': evaluator.validation_verbs_per_caption,
        'train_nouns_per_caption': evaluator.train_nouns_per_caption,
        'validation_nouns_per_caption': evaluator.validation_nouns_per_caption,
        'train_unique_caption_counts': evaluator.train_unique_caption_counts,
        'validation_unique_caption_counts': evaluator.validation_unique_caption_counts,
        'within_sample_train_caption_novelty': evaluator.within_sample_train_caption_novelty,
        'within_sample_validation_caption_novelty': evaluator.within_sample_validation_caption_novelty,
        'between_sample_train_caption_novelty': evaluator.between_sample_train_caption_novelty,
        'between_sample_validation_caption_novelty': evaluator.between_sample_validation_caption_novelty,
        'train_within_sample_score': evaluator.train_within_sample_score(),
        'validation_within_sample_score': evaluator.validation_within_sample_score(),
        'train_within_sample_score_masked': evaluator.train_within_sample_score_masked(),
        'validation_within_sample_score_masked': evaluator.validation_within_sample_score_masked(),
    }

    nss = max(outputs['validation_unique_caption_counts'])
    logging.info(f'Computing {nss} iterated scores.')
    for i in range(1, nss):
        outputs[f'validation_within_sample_score_gt_limit_{i}'] = evaluator.validation_within_sample_score(gt_limit=i)

    end_time = time.time()
    logging.info(f'Evaluation took {end_time - start_time} seconds.')

    with open((os.path.basename(FLAGS.json) + '.pkl'), 'wb') as f:
        pickle.dump(outputs, f)

if __name__ == '__main__':
    flags.mark_flag_as_required('json') # Require a JSON path

    app.run(main)
