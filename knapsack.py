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
import idf
import logging
logger = logging.getLogger(__name__)


def score_sentence_knapsack(*l_sents):
    kp = comp_model.Comp_we(WE_MODEL, l_sents)
    kp.prepare()
    ocs_ikj = [[[kp.d_concept[c] for c in sen] for sen in kp.s_ik[doc]]
               for doc in range(len(kp.c_ij))]
    # print(ocs_ikj)
    # dict_idf = idf.generate_idf('u08')
    dict_idf = comp_model.reuters_idf_dict(l_sents, "reuters")
    for concept in kp.w_ij[0].keys():
        kp.w_ij[0][concept] = kp.w_ij[0][concept]*dict_idf[concept]
    for concept in kp.w_ij[1].keys():
        kp.w_ij[1][concept] = kp.w_ij[1][concept]*dict_idf[concept]
    
    id_summary = []
    for i in range(len(l_sents)): 
        print('Summary ' + str(i))
        id_summary.append(knapsack(i, 100, kp.c_ij, ocs_ikj, kp.w_ij, kp.u_jk,
                                  kp.s_ik, kp.l_ik))
    # id_summary = knapsack(200, kp.c_ij, ocs_ikj, kp.w_ij, kp.u_jk,
                          # kp.s_ik, kp.l_ik)

    # print(id_summary)
    summary = []
    for j in range(len(id_summary)):
        summary.append([l_sents[j][i[1]] for i in id_summary[j] if i[0] == j])
    # summary_A = [l_sents[0][i[1]] for i in id_summary if i[0] == 0]
    # summary_B = [l_sents[1][i[1]] for i in id_summary if i[0] == 1]

    return summary[0], summary[1]


# A Dynamic Programming based Python Program for 0-1 Knapsack problem
# Returns the maximum value that can be put in a knapsack of capacity W
def knapsack(i, sumSize, c_ij, ocs_ikj, w_ij, u_jk, s_ik, l_ik):
    """knapsack

    :param l_sents:
    :param sumSize:
    :param ocs_ikj:
    :param w_ij:
    :param u_jk:
    :param s_ik:
    :param l_ik:
    """
    lambd = 0.55
    # K = cube of (value, summary (as a list of tuple (as coordinate of
    # sentence)))
    K = [[[0, []] for w in range(sumSize + 1)]
         for s in range(len(s_ik[i]) + 1)]

    for s in range(len(s_ik[i]) + 1):
        d = i
        sen = s-1
        logger.debug((d, sen))
        for w in range(sumSize + 1):
            if s == 0 or w == 0:
                pass
            elif l_ik[d][sen] <= w:
                current_sum = list(K[s-1][w-l_ik[d][sen]][1])
                current_sum.append((d, sen))
                value = obj(lambd, c_ij, ocs_ikj, w_ij, u_jk, current_sum)
                if value > K[s-1][w][0]:
                    K[s][w][0] = value
                    K[s][w][1] = current_sum
                else:
                    # print(value)
                    # print(current_sum)
                    K[s][w] = K[s-1][w]
            else:
                K[s][w] = K[s-1][w]
    return K[len(K)-1][sumSize][1]

# A Dynamic Programming based Python Program for 0-1 Knapsack problem
# Returns the maximum value that can be put in a knapsack of capacity W
def bi_knapsack(sumSize, c_ij, ocs_ikj, w_ij, u_jk, s_ik, l_ik):
    """knapsack

    :param l_sents:
    :param sumSize:
    :param ocs_ikj:
    :param w_ij:
    :param u_jk:
    :param s_ik:
    :param l_ik:
    """
    lambd = 0.55
    # K = cube of (value, summary (as a list of tuple (as coordinate of
    # sentence)))
    K_1 = [[[0, []] for w in range(sumSize + 1)]
         for s in range(len(s_ik[0]) + 1)]
    K_2 = [[[0, []] for w in range(sumSize + 1)]
         for s in range(len(s_ik[1]) + 1)]
    K = []
    K.append(K_1)
    K.append(K_2)

    max_size = max(len(s_ik[0]), len(s_ik[1]))

    for w in range(sumSize + 1):
        for size in range(len(max_size)):
            for s_0 in range(min(size, len(s_ik[0]))):
                cor = 0
                sen = s_0-1
                if s_0 == 0 or w == 0:
                    pass
                elif l_ik[cor][sen] <= w:
                    current_sum = list(K_0[sen-1][w-l_ik[cor][sen]][1])
                    current_sum.append((cor, sen))
                    sum_1 = list(K_1[min(size, len(s_ik[1]))][w]
                    value = obj(lambd, c_ij, ocs_ikj, w_ij, u_jk, current_sum,
                               sum_1)
                    if value > K_0[s-1][w][0]:
                        K_0[s][w][0] = value
                        K_0[s][w][1] = current_sum
                    else:
                        K_0[s][w] = K_0[s-1][w]
                else:
                    K[s][w] = K[s-1][w]
                    
            for s_1 in range(min(size, len(s_ik[1]))):
                if s_1 == 0 or w == 0:
                    pass

    for s in range(len(s_ik[0]) + len(s_ik[1]) + 1):
        c = 0 if s < limit_c0 else 1
        sen = s-1 if c == 0 else s-limit_c0
        logger.debug((c, sen))
        for w in range(sumSize + 1):
            if s == 0 and c == 0 or w == 0:
                pass
            elif l_ik[c][sen] <= w:
                current_sum = list(K[s-1][w-l_ik[c][sen]][1])
                current_sum.append((c, sen))
                value = obj(lambd, c_ij, ocs_ikj, w_ij, u_jk, current_sum)
                if value > K[s-1][w][0]:
                    K[s][w][0] = value
                    K[s][w][1] = current_sum
                else:
                    K[s][w] = K[s-1][w]
            else:
                K[s][w] = K[s-1][w]
    return K[len(K)-1][sumSize][1]


def obj(lambd, c_ij, ocs_ikj, w_ij, u_jk, summary):
    # print(type(w_ij))
    # print(type(w_ij[0]))
    # print(w_ij[0])
    comp = 0.
    rep = 0.
    lc = []
    # for _ in range(nbDoc):
        # lc.append(set())
    lc_0 = set()
    lc_1 = set()
    for sen in summary:
        # sen = (i, k)
        # print(sen)
        if sen[0] == 0:
            for concept in ocs_ikj[0][sen[1]]:
                lc_0.add(concept[0])
                if concept[1] is not None:
                    lc_1.add(concept[1])
        elif sen[0] == 1:
            for concept in ocs_ikj[1][sen[1]]:
                lc_1.add(concept[1])
                if concept[0] is not None:
                    lc_0.add(concept[0])
        else:
            pass
    # print(lc_0)
    # print(lc_1)
    t_rep = False
    # for i, lc_i in enumerate(lc):
        # for c_i in lc_i:
            # rep += w_ij[i][c_ij[i][c_i]]
            # for j in range(i, len(lc)):
                # for c_j in lc[j]:
                    # if (c_i, c_j) in u_jk:
                        # comp += u_jk[(c_0, c_1)]
    for c_0 in lc_0:
        rep += w_ij[0][c_ij[0][c_0]]
        for c_1 in lc_1:
            if not t_rep:
                rep += w_ij[1][c_ij[1][c_1]]
            if (c_0, c_1) in u_jk:
                # logger.info("COMP!!")
                comp += u_jk[(c_0, c_1)]
        t_rep = True
    return lambd*comp + (1-lambd)*rep
