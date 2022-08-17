# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Heonjin Kwon

import json
import logging
from typing import List, Union, Tuple

import pandas as pd
from gensim.models import Word2Vec, KeyedVectors
from anytree import Node, RenderTree
from sklearn.preprocessing import minmax_scale
from tqdm.auto import tqdm

MAX_WORDS_IN_BATCH = 10000
SENTIMENTAL_DICTIONARY_PATH = '../data/SentiWord_info.json'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Doc2Category(Word2Vec):
    def __init__(
            self, senti_dict_path: str = SENTIMENTAL_DICTIONARY_PATH, sentences=None, corpus_file=None, vector_size=100,
            alpha=0.025, window=5, min_count=5, max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
            sg=0, hs=0, negative=5, ns_exponent=0.75, cbow_mean=1, hashfxn=hash, epochs=5, null_word=0,
            trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH, compute_loss=False, callbacks=(),
            comment=None, max_final_vocab=None, shrink_windows=True,
    ):
        super().__init__(
            sentences=sentences, corpus_file=corpus_file, vector_size=vector_size, alpha=alpha,
            window=window, min_count=min_count, max_vocab_size=max_vocab_size, sample=sample, seed=seed,
            workers=workers, min_alpha=min_alpha, sg=sg, hs=hs, negative=negative, ns_exponent=ns_exponent,
            cbow_mean=cbow_mean, hashfxn=hashfxn, epochs=epochs, null_word=null_word, trim_rule=trim_rule,
            sorted_vocab=sorted_vocab, batch_words=batch_words, compute_loss=compute_loss,
            callbacks=callbacks, comment=comment, max_final_vocab=max_final_vocab,
            shrink_windows=shrink_windows
        )
        self.word_roots = None
        self.category = None
        self.is_tagged = False

        if not hasattr(self, 'wv'):  # set unless subclass already set (eg: FastText)
            self.wv = KeyedVectors(vector_size)

        with open(file=senti_dict_path,
                  mode='rt',
                  encoding='UTF8') as f:
            self.senti = pd.DataFrame.from_dict(json.load(f))
            self.senti['polarity'] = self.senti['polarity'].apply(int)

    def init_tree(self, categories: dict):
        roots = tuple(Node(name=k, data=k) for k in categories.keys())
        for root in roots:
            for word in categories[root.name]:
                Node(name=word, data=word, parent=root)
        self.word_roots = roots

    def print_trees(self):
        for root in self.word_roots:
            for pre, fill, node in RenderTree(root):
                logger.info("%s%s" % (pre, node.name))

    def print_tree(self, category):
        for root in self.word_roots:
            if root.name is category:
                for pre, fill, node in RenderTree(root):
                    logger.info("%s%s" % (pre, node.name))
                return

    def make_tree(self,
                  parents: Union[List[Node], Tuple[Node]],
                  width: int,
                  depth: int) -> None:
        if parents[0].depth >= depth:
            return

        if parents[0].is_leaf:
            for parent in parents:
                for word in tuple(map(lambda x: x[0], self.wv.most_similar(str(parent.name), topn=width))):
                    Node(name=word, data=word, parent=parent)

        for parent in parents:
            self.make_tree(parent.children, width, depth)

    def __count_root_words(self):
        __word_count_dict = dict()
        for root in self.word_roots:
            words = [child.data for child in root.descendants]
            words_count = {word: words.count(word) for word in set(words)}
            __word_count_dict[root.name] = words_count
        return __word_count_dict

    def tag(self, categories: dict, width: int = 3, depth: int = 3):
        self.init_tree(categories=categories)
        self.make_tree(self.word_roots, width=width, depth=depth)
        __word_count_dict = self.__count_root_words()

        # word counts to senti dict
        for category, words in __word_count_dict.items():
            self.senti[category] = 0
            for word, count in tqdm(words.items(), desc=category):
                condition = (self.senti['word'] == word) | (self.senti['word_root'] == word)
                self.senti.loc[condition, category] += count

        # scaling
        self.senti.loc[:, categories.keys()] = minmax_scale(self.senti.loc[:, categories.keys()], axis=1)

        # multiply polarity
        for idx in tqdm(range(len(self.senti)), desc='polarity'):
            self.senti.loc[idx, categories.keys()] *= float(self.senti.loc[idx, 'polarity'])

        self.category = categories.keys()
        self.is_tagged = True

    def score_review(self, tokenized_review: str) -> dict:
        """
        Scoring Reviews By Category.
        """
        if self.is_tagged is not True:
            raise Exception('Sentimental Dictionary of Review Scorer is not tagged yet. '
                            'Use method ReviewScorer.tag().')

        _scored = self.senti.loc[self.senti.word.isin(tokenized_review)].copy()

        return {category: sum(_scored[category]) for category in self.category}
