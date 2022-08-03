# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Heonjin Kwon
import json
from typing import List, Union

import pandas as pd
from gensim.models import Word2Vec


class ReviewScorer(object):
    def __init__(self,
                 tokens: Union[List[List[str]], pd.DataFrame],
                 senti_dict_path: str):
        self.tokens = tokens
        self.model = None
        with open(file=senti_dict_path,
                  mode='rt',
                  encoding='UTF8') as f:
            self.senti = pd.DataFrame.from_dict(json.load(f))

    def get_similar_words_indexes(self,
                                  words: List[str],
                                  topn: int):
        words_set = set(w[0] for word in words for w in self.model.wv.most_similar(word, topn=topn))
        words_set.update(words)
        result = self.senti['word'].isin(words_set) | self.senti['word_root'].isin(words_set)
        return result

    def tag_senti_dict(self,
                       taste_words: List[str],
                       price_words: List[str],
                       atmosphere_words: List[str],
                       service_words: List[str]):
        categories = {"taste": taste_words,
                      "price": price_words,
                      "atmosphere": atmosphere_words,
                      "service": service_words}

        for category, words in categories.items():
            self.senti[category] = False
            self.senti.loc[get_similar_words_indexes(words=words,
                                                     senti_df=self.senti,
                                                     model=model), category] = True

    def score_review(self,
                     review: str) -> dict:
        pass
