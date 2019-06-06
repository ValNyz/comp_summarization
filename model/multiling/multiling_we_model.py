# -*- coding: utf-8 -*-

"""

__author__ : Valentin Nyzam
"""
import os
import re
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors

class MultilingModels(object):

    def __init__(self, models_path, models_name_convention, l_sents):
        """
        Load models based on language in present in l_sents.

        Parameters
        ----------
        models_path : str
            Path to the models folder.
        models_name : str
            Models name convention (ex: LANG_cbow.bin)
        l_sents : list of browse_corpus.Sentence
            List of browse_corpus.Sentence.
        """
        self.models = {}
        for sent in l_sents:
            if sent.language not in self.models:
                model_name = re.sub('LANG', sent.language,
                                    models_name_convention)
                self.models[sent.language] = WordEmbeddingsKeyedVectors.load(os.path.join(models_path, model_name))

    def __getitem__(self, language):
        return self.models[language]
