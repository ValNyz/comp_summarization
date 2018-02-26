# -*- coding: utf-8 -*-

"""
Comparative summarization method using dynamic programming solution to Knapsack
problem
Inspired by :
    Comparative News Summarization Using Linear Programming (Huang et al, 2011)
__author__ : Valentin Nyzam
"""
import ilp
import logging
logger = logging.getLogger(__name__)


def score_sentence_knapsack(*l_sents):
    kp = ilp.Ilp_we('testWiki', l_sents)
    kp.prepare()
    ocs_ikj = [[[1 if j in k else 0 for j in kp.c_ij[i]] for k in kp.s_ik[i]]
               for i in range(len(kp.c_ij))]
    summary = knapsack(200, ocs_ikj, kp.w_ij, kp.u_jk, kp.s_ik, kp.l_ik)

    for sent in summary:
        print(l_sents[sent[0]][sent[1]])

    return None


# A Dynamic Programming based Python Program for 0-1 Knapsack problem
# Returns the maximum value that can be put in a knapsack of capacity W
def knapsack(sumSize, ocs_ikj, w_ij, u_jk, s_ik, l_ik):
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
    K = [[[[0, []] for w in range(sumSize + 1)]
          for s1 in range(len(s_ik[1]) + 1)]
         for s0 in range(len(s_ik[0]) + 1)]

    # Build table K[][] in bottom up manner
    for s0 in range(len(s_ik[0]) + 1):
        for s1 in range(len(s_ik[1]) + 1):
            for w in range(sumSize + 1):
                if s0 == 0 or s1 == 0 or w == 0:
                    # K[s0][s1][w] = (0, [])
                    pass
                elif l_ik[0][s0-1] + l_ik[1][s1-1] <= w:
                    current_sum = list(K[s0][s1-1]
                                        [w-l_ik[0][s0-1]-l_ik[1][s1-1]]
                                        [1])
                    current_sum.append((0, s0-1))
                    current_sum.append((1, s1-1))
                    value = obj(0.55, ocs_ikj, w_ij, u_jk, current_sum)
                    if value > K[s0][s1-1][w][0]:
                        K[s0][s1][w][0] = value
                        K[s0][s1][w][1] = current_sum
                    else:
                        K[s0][s1][w] = K[s0][s1-1][w]
                else:
                    K[s0][s1][w] = K[s0][s1-1][w]
    print(K)
    return K[len(s_ik[0])+1][len(s_ik[1]+1)][sumSize+1]


def obj(lambd, ocs_ikj, w_ij, u_jk, summary):
    comp = 0.
    rep = 0.
    lc_0 =set()
    lc_1 = set()
    for sen in summary:
        # sen = (i, k)
        if sen[0] == 0:
            lc_0.update([j for j in range(len(ocs_ikj[0][sen[1]]))])
        elif sen[0] == 1:
            lc_1.update([j for j in range(len(ocs_ikj[1][sen[1]]))])
        else:
            pass
    t_rep = False
    for c_0 in lc_0:
        rep += w_ij[0][c_0]
        for c_1 in lc_1:
            if not t_rep:
                rep += w_ij[1][c_1]
            comp += u_jk[c_0][c_1]
        t_rep = True
    return lambd*comp + (1-lambd)*rep
