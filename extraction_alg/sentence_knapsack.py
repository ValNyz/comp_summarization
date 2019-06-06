# -*- coding: utf-8 -*-

"""
Comparative summarization method using dynamic programming solution to Knapsack
problem
Inspired by :
    Comparative News Summarization Using Linear Programming (Huang et al, 2011)
__author__ : Valentin Nyzam
"""

# from model import comp_model
# from globals import WE_MODEL

from random import shuffle
from operator import itemgetter

from nltk.util import ngrams

import time

import logging
logger = logging.getLogger(__name__)


def score_sentence_knapsack2(model, lambd, l_sents):
    dictionary = model.dictionary
    d_word_tfidf = model.d_word_tfidf
    d_sen_score = model.d_sen_score
    d_sen_sim = model.d_sen_sim
    d_id_sen = model.d_id_sents_corpus

    id_summary = bi_knapsack(100, lambd, model, l_sents)
    s = sorted(id_summary, key=itemgetter(0,1))

    summary_A = [l_sents[0][i[1]] for i in s if i[0] == 0]
    summary_B = [l_sents[1][i[1]] for i in s if i[0] == 1]

    return summary_A, summary_B


# A Dynamic Programming based Python Program for 0-1 Knapsack problem
# Returns the maximum value that can be put in a knapsack of capacity W
def bi_knapsack(sumSize, lambd, model, l_sents):
    """knapsack

    """
    logger.info('Launching Knapsack extraction algorithm.')
    l_sen = []
    k = 0
    for i, corpus in enumerate(l_sents):
        for j, sen in enumerate(corpus):
            l_sen.append((i, j, k, sen))
            k += 1
    # print(l_sen)
    # exit()
    # shuffle(l_sen)

    # K = cube of (value, summary (as a list of tuple (as coordinate of
    # sentence)))
    K = [[[[0, []] for w2 in range(sumSize + 1)]
          for w1 in range(sumSize + 1)]
         for s in range(len(l_sen) + 1)]

    w_0 = 0
    w_1 = 0
    boolean = True

    start = time.process_time()
    while w_0 < sumSize + 1 and w_1 < sumSize + 1:
    # for w_0 in range(sumSize + 1):
        # for w_1 in range(sumSize + 1):
        for i in range(len(l_sen) + 1):
            if i == 0 or w_0 == 0 or w_1 == 0:
                pass
            cor = l_sen[i-1][0]
            sen = l_sen[i-1][1]
            id_sen = l_sen[i-1][2]
            sentence = l_sen[i-1][3]
            if cor == 0 and len(sentence) <= w_0:
                current_sum = list(K[i-1][w_0-len(sentence)][w_1][1])
                current_sum.append(l_sen[i-1])
                value = obj(lambd, model, current_sum)
                if value >= K[i-1][w_0][w_1][0]:
                    # print('value %f' % value)
                    # print('%i %i' % (w_0, w_1))
                    K[i][w_0][w_1][0] = value
                    K[i][w_0][w_1][1] = current_sum
                else:
                    K[i][w_0][w_1] = K[i-1][w_0][w_1]
            elif cor == 1 and len(sentence) <= w_1:
                current_sum = list(K[i-1][w_0][w_1-len(sentence)][1])
                current_sum.append(l_sen[i-1])
                value = obj(lambd, model, current_sum)
                if value >= K[i-1][w_0][w_1][0]:
                    K[i][w_0][w_1][0] = value
                    K[i][w_0][w_1][1] = current_sum
                else:
                    K[i][w_0][w_1] = K[i-1][w_0][w_1]
            else:
                K[i][w_0][w_1] = K[i-1][w_0][w_1]
        if boolean:
            w_0 += 1
        else:
            w_1 += 1
        boolean = not boolean
        logger.info("Iteration " + str(w_0) + " : " +
                    str(K[len(K)-1][w_0][w_1][0]))
        if w_0 < 10:
            print(K[len(K)-1][w_0][w_1][1])
            for sen in K[len(K)-1][w_0][w_1][1]:
                print(len(sen[3]))
                print(sen[3])
    logger.info('Knapsack execution time: ' + str(time.process_time() - start))
    # print(K[len(l_sen)-1])
    return K[len(K)-1][sumSize][sumSize][1]


def obj(lambd, model, summary):
    d_id_sen = model.d_id_sents_corpus

    ls_1 = []
    ls_2 = []
    for sen in summary:
        if sen[0] == 0:
            ls_1.append(d_id_sen[(sen[0], sen[1])])
        else:
            ls_2.append(d_id_sen[(sen[0], sen[1])])

    rep = informative_1(model, summary)
    if rep < 0:
        rep = 0

    comp = 0
    for s1 in ls_1:
        for s2 in ls_2:
            comp += model.d_sen_sim[(s1, s2)]

    score = lambd*comp + (1-lambd)*rep

    return score / len(summary)


def informative_1(model, summary):
    rep = 0
    for sen in summary:
        rep += model.d_sen_score[sen[2]]
    return rep


def informative_2(model, summary):
    diff = model.max_score - model.min_score
    rep = 0
    set_words = set()
    for sen in summary:
        temp = 0
        for word in sen[3]:
            if word not in set_words:
                set_words.add(word)
                try:
                    temp += model.d_word_tfidf[sen[0]][model.dictionary.token2id[word]]
                except:
                    pass
        rep += (temp - model.min_score)/diff
    return rep


def informative_3(model, summary):
    rep = 0
    set_words = set()
    nb_words = 0
    for sen in summary:
        for word in sen:
            if word not in set_words:
                set_words.add(word)
                try:
                    rep += model.d_word_tfidf[sen[0]][model.dictionary.token2id[word]]
                except:
                    pass
            nb_words += 1
    return rep / nb_words


def informative_4(model, summary):
    rep = 0

    sum_A = []
    sum_B = []
    nb_words = 0
    for sen in summary:
        nb_words += len(sen)
        if sen[0] == 0:
            sum_A.extend(sen[3])
        else:
            sum_B.extend(sen[3])

    if len(sum_A) != 0:
        sum_A_tf_idf = dict(model.tfidf_model[model.dictionary.doc2bow(sum_A)])
        for word in sum_A:
            try:
                rep += sum_A_tf_idf[model.dictionary.token2id[word]]
            except:
                pass
    if len(sum_B) != 0:
        sum_B_tf_idf = dict(model.tfidf_model[model.dictionary.doc2bow(sum_B)])
        for word in sum_B:
            try:
                rep += sum_B_tf_idf[model.dictionary.token2id[word]]
            except:
                pass

    return rep / nb_words
