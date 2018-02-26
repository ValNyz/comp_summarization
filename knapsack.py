# -*- coding: utf-8 -*-

"""
Comparative summarization method using dynamic programming solution to Knapsack
problem
Inspired by :
    Comparative News Summarization Using Linear Programming (Huang et al, 2011)
__author__ : Valentin Nyzam
"""
import os
import gensim.models
from itertools import product, chain
from collections import Counter
from nltk.util import ngrams
from nltk.corpus import wordnet as wn
import numpy as np
from scipy.spatial.distance import cosine
import pulp
from pulp import lpSum
import logging
logger = logging.getLogger(__name__)


def score_sentence_knapsack(*l_sents):
    return None


def generate_knapsack_problem(c_ij, w_ij, u_jk, s_ik, l_ik):
    """generate_ilp_problem
    :param c_ij: list[list[concept]]: list of concept: i #doc, j #concept
    :param w_ij: list[dict{concept:weight}]: dict of concept:weight for each
    document
    :param u_jk: dict{concept_pair:weight}: dict of concept_pair:weight
    :param s_ik: list[list[list[concept]]]: list doc as a list of sentence as a
    list of concept
    """
    ocs_ikj = [[[0 if j in k else 1 for j in c_ij[i]] for k in s_ik[i]] for
               i in range(len(c_ij))]


# A Dynamic Programming based Python Program for 0-1 Knapsack problem
# Returns the maximum value that can be put in a knapsack of capacity W
def knapsack(sumSize, c_ij, w_ij, u_jk, s_ik, l_ik):
    """knapsack

    :param sumSize:
    :param c_ij:
    :param w_ij:
    :param u_jk:
    :param s_ik:
    :param l_ik:
    """
    # K = cube of (value, summary (as a list of tuple (as coordinate of
    # sentence)))
    K = [[[(0, []) for w in range(sumSize+1)]
          for s1 in range(len(s_ik[1]) + 1)]
         for s0 in range(len(s_ik[0]) + 1)]

    # Build table K[][] in bottom up manner
    for s0 in range(len(s_ik[0])):
        for s1 in range(len(s_ik[1])):
            for w in range(sumSize+1):
                if s0 == 0 or s1 == 0 or w == 0:
                    K[s0][s1][w] = (0, [])
                elif l_ik[0][s0]+l_ik[1][s1-1] <= w and l_ik[1][s1] <= w:
                    value = obj(K[s0][s1][w][1])
                    if value > K[s0][s1][w][0]:
                        val[i-1] + K[i-1][w-wt[i-1]]
                        K[s0][s1][w][1].extend(K[s0][s1-1][w][1])
                        K[s0][s1][w][1].append((0, s0))
                        K[s0][s1][w][1].append((1, s1))
                        K[s0][s1][w][0] = value
                    else:
                        K[s0][s1][w] = K[s0][s1-1][w]
                    K[s0][s1][w] = max(objective_func[i-1] + K[i-1][w-wt[i-1]],  K[i-1][w])
                else:
                    K[i][w] = K[i-1][w]

    return K[n][sumSize]


def obj(summary):
    pass
