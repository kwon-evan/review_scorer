# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Heonjin Kwon

import json
import logging
from typing import List, Union

import pandas as pd
from gensim.models import Word2Vec

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s - %(message)s')


class ReviewScorer(object):
    def __init__(self,
                 tokens: Union[List[List[str]], pd.DataFrame],
                 senti_dict_path: str):
        self.category_list = None
        self.is_tagged = False
        print('ReviewScorer Initializing ..')
        self.model = Word2Vec(sentences=tokens,
                              vector_size=100,
                              window=5,
                              min_count=2,
                              workers=4,
                              hs=1)
        print('Done.')

        with open(file=senti_dict_path,
                  mode='rt',
                  encoding='UTF8') as f:
            self.senti = pd.DataFrame.from_dict(json.load(f))
            self.senti['polarity'] = self.senti['polarity'].apply(int)

    def save(self, *args, **kwargs):
        self.model.save(*args, **kwargs)
        
    def load(self, *args, **kwargs):
        self.model.load(*args, **kwargs)

    def train(self,
              tokens: Union[List[List[str]], pd.DataFrame]):
        self.model.train(tokens)

    def get_similar_words_indexes(self,
                                  words: List[str],
                                  topn: int = 100):
        words_set = set(w[0] for word in words for w in self.model.wv.most_similar(word, topn=topn))
        words_set.update(words)
        result = self.senti['word'].isin(words_set) | self.senti['word_root'].isin(words_set)
        return result

    def tag_senti_dict(self, categories: dict, topn: int = 100):
        self.category_list = categories.keys()
        for category, words in categories.items():
            logger.info(f'tagging {category}')
            self.senti[category] = False
            self.senti.loc[self.get_similar_words_indexes(words=words, topn=topn), category] = True
        self.is_tagged = True

    def score_review(self,
                     tokenized_review: str) -> dict:
        if self.is_tagged is not True:
            raise Exception('Sentimental Dictionary of Review Scorer is not tagged yet. '
                            'Use method tag_senti_dict.\n')

        _scored = self.senti.loc[self.senti.word.isin(tokenized_review)].copy()
        result = {
            category_name: _scored.loc[_scored[category_name], 'polarity'].sum()
            for category_name in self.category_list
        }
        
        return result
