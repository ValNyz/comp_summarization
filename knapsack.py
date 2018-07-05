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
    ocs_ikj = [[[kp.d_concept[c] for c in k] for k in kp.s_ik[i]]
               for i in range(len(kp.c_ij))]
    # print(ocs_ikj)
    # dict_idf = idf.generate_idf('u08')
    dict_idf = comp_model.reuters_idf_dict(l_sents, "reuters")
    for concept in kp.w_ij[0].keys():
        kp.w_ij[0][concept] = kp.w_ij[0][concept]*dict_idf[concept]
    for concept in kp.w_ij[1].keys():
        kp.w_ij[1][concept] = kp.w_ij[1][concept]*dict_idf[concept]
    
    # summaries = []
    # for doc in l_sents: 
        # summaries.append(knapsack(doc, 100, kp.c_ij, ocs_ikj, kp.w_ij, kp.u_jk,
                                  # kp.s_ik, kp.l_ik))
    id_summary = knapsack(l_sents, 200, kp.c_ij, ocs_ikj, kp.w_ij, kp.u_jk,
                          kp.s_ik, kp.l_ik)

    summary_A = [l_sents[0][i[1]] for i in id_summary if i[0] == 0]
    summary_B = [l_sents[1][i[1]] for i in id_summary if i[0] == 1]

    return summary_A, summary_B


# A Dynamic Programming based Python Program for 0-1 Knapsack problem
# Returns the maximum value that can be put in a knapsack of capacity W
def knapsack(l_sents, sumSize, c_ij, ocs_ikj, w_ij, u_jk, s_ik, l_ik):
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
         for s in range(len(s_ik[0]) + len(s_ik[1]) + 1)]
    limit_c0 = len(s_ik[0])+1
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
    lc_0 = set()
    lc_1 = set()
    for sen in summary:
        # sen = (i, k)
        # print(sen)
        if sen[0] == 0:
            lc_0.update([j[0] for j in ocs_ikj[0][sen[1]]])
        elif sen[0] == 1:
            lc_1.update([j[1] for j in ocs_ikj[1][sen[1]]])
        else:
            pass
    t_rep = False
    for c_0 in lc_0:
        # print(type(w_ij[0][c_ij[0][c_0]]))
        rep += w_ij[0][c_ij[0][c_0]]
        for c_1 in lc_1:
            if not t_rep:
                rep += w_ij[1][c_ij[1][c_1]]
            if (c_0, c_1) in u_jk:
                comp += u_jk[(c_0, c_1)]
        t_rep = True
    return lambd*comp + (1-lambd)*rep
