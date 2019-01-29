# -*- coding: utf-8 -*-

"""
Comparative summarization method using dynamic programming solution to Knapsack
problem
Inspired by :
    Comparative News Summarization Using Linear Programming (Huang et al, 2011)
__author__ : Valentin Nyzam
"""
from model import comp_model
from globals import WE_MODEL
from random import shuffle

import time

import logging
logger = logging.getLogger(__name__)

def score_sentence_knapsack2(model, threshold, *l_sents):
    d_sen_score = model.d_sen_score
    d_sen_sim = model.d_sen_sim
    d_id_sen = model.d_id_sents_corpus

    id_summary = bi_knapsack(100, d_id_sen, d_sen_score, d_sen_sim, l_sents)

    summary_A = [l_sents[0][i[1]] for i in id_summary if i[0] == 0]
    summary_B = [l_sents[1][i[1]] for i in id_summary if i[0] == 1]

    return summary_A, summary_B

# A Dynamic Programming based Python Program for 0-1 Knapsack problem
# Returns the maximum value that can be put in a knapsack of capacity W
def bi_knapsack(sumSize, d_id_sen, d_sen_score, d_sen_sim, l_sents):
    """knapsack

    """
    l_sen = []
    for i, corpus in enumerate(l_sents):
        for j, sen in enumerate(corpus):
            l_sen.append((i, j, sen))
    shuffle(l_sen)

    lambd = 0.55
    # K = cube of (value, summary (as a list of tuple (as coordinate of
    # sentence)))
    K = [[[[0, []] for w2 in range(sumSize + 1)]
          for w1 in range(sumSize + 1)]
         for s in range(len(l_sen) + 1)]

    s_0 = 0
    s_1 = 0

    start = time.process_time()
    for w_0 in range(sumSize + 1):
        for w_1 in range(sumSize + 1):
            for i in range(len(l_sen) + 1):
                if i == 0 or w_0 == 0 or w_1 == 0:
                    pass
                cor = l_sen[i-1][0]
                sen = l_sen[i-1][1]
                sentence = l_sen[i-1][2]
                if cor == 0 and sentence.len <= w_0:
                    current_sum = list(K[i-1][w_0-sentence.len][w_1][1])
                    current_sum.append(l_sen[i-1])
                    value = obj(lambd, d_id_sen, d_sen_score, d_sen_sim, current_sum)
                    if value > K[i-1][w_0][w_1][0]:
                        # print(value)
                        # print(current_sum)
                        K[i][w_0][w_1][0] = value
                        K[i][w_0][w_1][1] = current_sum
                    else:
                        K[i][w_0][w_1] = K[i-1][w_0][w_1]
                elif cor == 1 and sentence.len <= w_1:
                    current_sum = list(K[i-1][w_0][w_1-sentence.len][1])
                    current_sum.append(l_sen[i-1])
                    value = obj(lambd, d_id_sen, d_sen_score, d_sen_sim, current_sum)
                    if value > K[i-1][w_0][w_1][0]:
                        # print(value)
                        # print(current_sum)
                        K[i][w_0][w_1][0] = value
                        K[i][w_0][w_1][1] = current_sum
                    else:
                        K[i][w_0][w_1] = K[i-1][w_0][w_1]
                else:
                    K[i][w_0][w_1] = K[i-1][w_0][w_1]
                # print(K[i][w_0][w_1])
        logger.info("Iteration " + str(w_0) + " : " +
                    str(K[len(K)-1][w_0][sumSize][0]))
    logger.info('Knapsack execution time : ' + str(time.process_time() - start))
    # print(K[len(l_sen)-1])
    return K[len(K)-1][sumSize][sumSize][1]

def obj(lambd, d_id_sen, d_sen_score, d_sen_sim, summary):
    ls_1 = []
    ls_2 = []
    for sen in summary:
        if sen[0] == 0:
            ls_1.append(d_id_sen[(sen[0], sen[1])])
            # ls_1.append(sen)
        else:
            ls_2.append(d_id_sen[(sen[0], sen[1])])
            # ls_2.append(sen)

    t_rep = True
    comp = 0
    rep = 0
    for s1 in ls_1:
        rep += d_sen_score[s1]
        for s2 in ls_2:
            comp += d_sen_sim[(s1, s2)]
            if t_rep:
                rep += d_sen_score[s2]
        if t_rep:
            t_rep = False
    return lambd*comp + (1-lambd)*rep
