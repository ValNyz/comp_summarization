# -*- coding: utf-8 -*-

"""
Word Mover's Distance using word embeddings
    From Word Embeddings To Document Distances (Kusner et al, 2015)
__author__ : Valentin Nyzam
"""
from itertools import product
from collections import defaultdict
from scipy.spatial.distance import euclidean
import threading
import pulp


lock = threading.Lock()


def word_mover_distance(first_sent_tokens, second_sent_tokens, wvmodel,
                        lpFile=None):
    prob = _word_mover_distance_probspec(first_sent_tokens, second_sent_tokens,
                                         wvmodel, lpFile=lpFile)
    return pulp.value(prob.objective)


def _word_mover_distance_probspec(sent1, sent2, wvmodel, lpFile=None):
    all_tokens = list(sent1 | sent2)
    with lock:
        wordvecs = {token: wvmodel[token] for token in all_tokens}

    buckets1 = _tokens_to_fracdict(sent1)
    buckets2 = _tokens_to_fracdict(sent2)

    T = pulp.LpVariable.dicts('T_matrix',
                              list(product(all_tokens, all_tokens)),
                              lowBound=0)

    prob = pulp.LpProblem('WMD', sense=pulp.LpMinimize)
    prob += pulp.lpSum([T[token1, token2]
                        * euclidean(wordvecs[token1], wordvecs[token2])
                        for token1, token2 in product(all_tokens, all_tokens)])
    for token2 in buckets2:
        prob += pulp.lpSum([T[token1, token2] for token1 in
                            buckets1]) == buckets2[token2]
    for token1 in buckets1:
        prob += pulp.lpSum([T[token1, token2] for token2 in
                            buckets2]) == buckets1[token1]

    if lpFile is not None:
        prob.writeLP(lpFile)

    prob.solve()

    return prob


def _tokens_to_fracdict(tokens):
    cntdict = defaultdict(lambda: 0)
    for token in tokens:
        cntdict[token] += 1
    totalcnt = sum(cntdict.values())
    return {token: float(cnt)/totalcnt for token, cnt in cntdict.items()}
