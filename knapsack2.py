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
    dict_idf = model.dict_idf
    d_sim_sen = model.d_sen_sim
    word_weight = {}
    for doc in l_sents:
        for sen in doc:
            for word in sen:
                if word in word_weight:
                    word_weight[word] += 1
                else:
                    word_weight[word] = 1

    for word in word_weight.keys():
        word_weight[word] = word_weight[word] * dict_idf[word]

    comp_weight = {}
    for pair, sim in d_sim_sen.items():
        i = pair[0]
        j = pair[1]
        weight = 0
        for w1 in l_sents[0][pair[0]]:
            weight += word_weight[w1]
        for w2 in l_sents[1][pair[1]]:
            weight += word_weight[w2]
        comp_weight[pair] = weight * sim

    id_summary = bi_knapsack(100, word_weight, comp_weight, l_sents)

    summary_A = [l_sents[0][i[1]] for i in id_summary if i[0] == 0]
    summary_B = [l_sents[1][i[1]] for i in id_summary if i[0] == 1]

    return summary_A, summary_B

# A Dynamic Programming based Python Program for 0-1 Knapsack problem
# Returns the maximum value that can be put in a knapsack of capacity W
def bi_knapsack(sumSize, word_weight, comp_weight, *l_sents):
    """knapsack

    """
    l_sen = []
    for i, doc in enumerate(l_sents):
        for j, s in enumerate(doc):
            l_sen.append((i, j, s))
    shuffle(l_sen)

    lambd = 0.55
    # K = cube of (value, summary (as a list of tuple (as coordinate of
    # sentence)))
    K = [[[ [0, []] for w2 in range(sumSize + 1)]
                        for w1 in range(sumSize + 1)]
         for s in range(len(l_sen) + 1)]

    s_0 = 0
    s_1 = 0

    for w_0 in range(sumSize + 1):
        logger.info("Iteration " + str(w_0))
        start = time.process_time()
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
                    value = obj(lambd, word_weight, comp_weight, current_sum)
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
                    value = obj(lambd, word_weight, comp_weight, current_sum)
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
                logger.info('Knapsack execution time : ' + str(time.process_time() - start))
    # print(K[len(l_sen)-1])
    return K[len(K)-1][sumSize][sumSize][1]

def obj(lmabd, word_weight, comp_weight, summary):
    l_s1 = []
    l_s2 = []
    for sen in summary:
        if sen[0] == 0:
            ls_1.append(sen)
        else:
            ls_2.append(sen)

    t_rep = True
    comp = 0
    rep = 0
    for s1 in ls_1:
        for w in s1[2]:
            rep += word_weight[w]
        for s2 in ls_2:
            comp += comp_weight[(s1[1], s2[1])]
            if t_rep:
                for w in s2[2]:
                    rep += word_weight[w]
        if t_rep:
            t_rep = False
    return lambd*comp + (1-lambd)*rep
