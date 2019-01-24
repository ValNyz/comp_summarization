# -*- coding: utf-8 -*-

"""
Lexrank modification of package lexrank using wmd similarity. Only use for scoring sentence
"""

from lexrank import LexRank
from lexrank.algorithms.power_method import stationary_distribution
import numpy as np

import logging
logger = logging.getLogger(__name__)

class LexRank_wmd(LexRank):
    def __init__(self):
        pass

    def get_summary(
            self,
            sentences,
            summary_size=1,
            threshold=.03,
            fast_power_method=True,
    ):
        pass

    def rank_sentences(
        self,
        length,
        d_sen_sim,
        threshold=.03,
        fast_power_method=True,
    ):
        if not (
            threshold is None or
            isinstance(threshold, float) and 0 <= threshold < 1
        ):
            raise ValueError(
                '\'threshold\' should be a floating-point number '
                'from the interval [0, 1) or None',
            )

        similarity_matrix = self._calculate_similarity_matrix(length, d_sen_sim)

        if threshold is None:
            markov_matrix = self._markov_matrix(similarity_matrix)

        else:
            markov_matrix = self._markov_matrix_discrete(
                similarity_matrix,
                threshold=threshold,
            )

        scores = stationary_distribution(
            markov_matrix,
            increase_power=fast_power_method,
            normalized=False,
        )

        return scores

    def _calculate_similarity_matrix(self, length, d_sen_sim):
        similarity_matrix = np.zeros([length] * 2)

        for i in range(length):
            for j in range(i+1, length):
                similarity = d_sen_sim[(i, j)]

                if similarity:
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity

        return similarity_matrix
