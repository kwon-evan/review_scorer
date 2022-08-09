# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Heonjin Kwon

import json
import logging
from typing import List, Union

import pandas as pd
from gensim.models import Word2Vec, KeyedVectors

MAX_WORDS_IN_BATCH = 10000
SENTIMENTAL_DICTIONARY_PATH = '../data/SentiWord_info.json'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ReviewScorer(Word2Vec):
    def __init__(
            self, senti_dict_path: str = SENTIMENTAL_DICTIONARY_PATH, sentences=None, corpus_file=None, vector_size=100,
            alpha=0.025, window=5, min_count=5, max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
            sg=0, hs=0, negative=5, ns_exponent=0.75, cbow_mean=1, hashfxn=hash, epochs=5, null_word=0,
            trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH, compute_loss=False, callbacks=(),
            comment=None, max_final_vocab=None, shrink_windows=True,
    ):
        super().__init__(sentences=sentences, corpus_file=corpus_file, vector_size=vector_size, alpha=alpha,
                         window=window, min_count=min_count, max_vocab_size=max_vocab_size, sample=sample, seed=seed,
                         workers=workers, min_alpha=min_alpha, sg=sg, hs=hs, negative=negative, ns_exponent=ns_exponent,
                         cbow_mean=cbow_mean, hashfxn=hashfxn, epochs=epochs, null_word=null_word, trim_rule=trim_rule,
                         sorted_vocab=sorted_vocab, batch_words=batch_words, compute_loss=compute_loss,
                         callbacks=callbacks, comment=comment, max_final_vocab=max_final_vocab,
                         shrink_windows=shrink_windows, )
        if not hasattr(self, 'wv'):  # set unless subclass already set (eg: FastText)
            self.wv = KeyedVectors(vector_size)
        self.category_list = None
        self.is_tagged = False
        with open(file=senti_dict_path,
                  mode='rt',
                  encoding='UTF8') as f:
            self.senti = pd.DataFrame.from_dict(json.load(f))
            self.senti['polarity'] = self.senti['polarity'].apply(int)

    def get_similar_words_indexes(self, words: List[str], topn: int = 100):
        words_set = set(w[0] for word in words for w in self.wv.most_similar(word, topn=topn))
        words_set.update(words)
        result = self.senti['word'].isin(words_set) | self.senti['word_root'].isin(words_set)
        return result

    def tag(self, categories: dict, topn: int = 100):
        self.category_list = categories.keys()
        for category, words in categories.items():
            logger.info(f'tagging {category}')
            self.senti[category] = False
            self.senti.loc[self.get_similar_words_indexes(words=words, topn=topn), category] = True
        self.is_tagged = True

    def score_review(self,
                     tokenized_review: str) -> dict:
        """
        Scoring Reviews By Category.
        """
        if self.is_tagged is not True:
            raise Exception('Sentimental Dictionary of Review Scorer is not tagged yet. '
                            'Use method ReviewScorer.tag.')

        _scored = self.senti.loc[self.senti.word.isin(tokenized_review)].copy()

        return {
            category_name: _scored.loc[_scored[category_name], 'polarity'].sum()
            for category_name in self.category_list
        }
