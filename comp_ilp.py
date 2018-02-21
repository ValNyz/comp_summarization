# -*- coding: utf-8 -*-

"""
Comparative summarization method using ILP
Comparative News Summarization Using Linear Programming (Huang et al, 2011)
__author__ : Valentin Nyzam
"""

from itertools import product
from collections import Counter
from nltk.util import ngrams
from nltk.corpus import wordnet as wn
import numpy as np
import pulp
from pulp import lpSum
import logging
logger = logging.getLogger(__name__)


def score_sentence_ilp(*l_sents):

    c_ij, s_ik, l_ik, tf_i = make_concept(*l_sents)
    print("Vocabulary size : " + str(len(c_ij[0]) + len(c_ij[1])))
    u_jk = make_concept_pair(c_ij, tf_i, relevance_wordnet)
    generate_ilp_problem(c_ij, tf_i, u_jk, s_ik, l_ik)
    return None


def generate_ilp_problem(c_ij, w_ij, u_jk, s_ik, l_ik):
    """generate_ilp_problem
    :param c_ij: list[list[concept]]: list of concept: i #doc, j #concept
    :param w_ij: list[dict{concept:weight}]: dict of concept:weight for each
    document
    :param u_jk: dict{concept_pair:weight}: dict of concept_pair:weight
    :param s_ik: list[list[list[concept]]]: list doc as a list of sentence as a
    list of concept
    """
    prob = pulp.LpProblem('summarize', pulp.LpMaximize)

    # oc_ij
    oc_ij = pulp.LpVariable.dicts('concepts', [(i, j) for i in range(len(c_ij))
                                               for j in range(len(c_ij[i]))],
                                  0, 1, pulp.LpBinary)
    # op_jk
    op_jk = pulp.LpVariable.dicts('pairs', [(j, k) for j in range(len(u_jk))
                                            for k in range(len(u_jk[j]))],
                                  0, 1, pulp.LpBinary)
    # ocs_ijk
    ocs_ijk = [[[0 if j in k else 1 for j in c_ij[i]] for k in s_ik[i]] for
               i in range(len(c_ij))]
    # os_ik
    os_ik = pulp.LpVariable.dicts('sentences',
                                  [(i, k) for i in range(len(c_ij))
                                   for k in range(len(s_ik))],
                                  0, 1, pulp.LpBinary)
    lambd = 0.55
    prob += lambd*lpSum([u_jk[j][k]*op_jk[(j, k)] for j in range(len(c_ij[0]))
                         for k in range(len(c_ij[1]))]) + \
            (1-lambd)*lpSum([w_ij[i][c_ij[i][j]]*oc_ij[(i, j)] for i in
                             range(len(c_ij)) for j in range(len(c_ij[i]))])
    # prob += pulp.lpSum([tf[w] * word_vars[w] for w in tf])
    for j in range(len(c_ij[0])):
        for k in range(len(c_ij[1])):
            prob += op_jk[(j, k)] <= oc_ij[(0, j)] and op_jk[(j, k)] <= \
                    oc_ij[(1, k)]
    for j in range(len(c_ij[0])):
        prob += [lpSum([op_jk[(j, k)] for k in range(len(c_ij[1]))])]
    for k in range(len(c_ij[1])):
        prob += [lpSum([op_jk[(j, k)] for j in range(len(c_ij[0]))])]
    for i in range(len(c_ij)):
        for k in range(len(s_ik[i])):
            for j in range(len(c_ij[i])):
                prob += oc_ij[(i, j)] >= ocs_ijk[i][j][k]*os_ik[(i, k)]
    for i in range(len(c_ij)):
        for j in range(len(c_ij[i])):
            prob += oc_ij[(i, j)] <= lpSum([ocs_ijk[i][j][k]*os_ik[(i, k)]
                                            for k in s_ik[i]])
    prob += lpSum([os_ik[(i, k)]*l_ik[i][k] for i in range(len(c_ij)) for
                  k in range(len(s_ik[i]))])
    # The problem data is written to an .lp file
    prob.writeLP("comp_ilp.lp")
    prob.solve()
    # The status of the solution is printed to the screen
    print("Status:", pulp.LpStatus[prob.status])
    # Each of the variables is printed with it's resolved optimum value
    for v in prob.variables():
        print(v.name, "=", v.varValue)
    return None


def make_concept(*l_sents, order=2):
    """make_concept

    :param *l_sents:
    :param order:
    :return list[list[concept]]: list of all concepts per doc
    :return list[list[list[concept]]]: list of doc as list of sentence as list
    of concepts
    :return list[dict{concept:tf}]
    """
    c_ij = []
    s_ik = []
    l_ik = []
    tf_i = []
    for i, doc in enumerate(l_sents):
        c_ij.append([])
        s_ik.append(set())
        l_ik.append([])
        for sentence in doc:
            sent = sentence.get_list_word_pos()
            temp = list(ngrams(sent, order))
            temp.extend(sent)
            s_ik[i].update(temp)
            l_ik[i].append(len(sent))
            c_ij[i].extend(temp)
        tf_i.append(Counter(c_ij[i]))
        c_ij[i] = set(c_ij[i])
    return c_ij, s_ik, l_ik, tf_i


def make_concept_pair(c_ij, w_ij, similarity, threshold=0.2):
    """make_dict_concept_pair

    :param c_ij:
    :param w_ij:
    :param similarity: similarity function: input(concept, concept):
        ouput(0<int<1)
    :param threshold:
    :return list[np.array]: size [j, k]
    """
    u_jk = []
    # build list pair concept
    for j, c_0 in enumerate(c_ij[0]):
        u_jk.append(np.zeros(len(c_ij[1]), float))
        for k, c_1 in enumerate(c_ij[1]):
            if similarity(c_0, c_1) > threshold:
                u_jk[j][k] = (w_ij[0][c_0]+w_ij[1][c_1])/2
            else:
                u_jk[j][k] = 0
    with open('pair_' + str(similarity) + '_' + str(threshold), 'w') as f:
        for j in range(len(u_jk)):
            for k in range(len(u_jk[j])):
                f.write(str(c_ij[0][j]) + '\t' + str(c_ij[1][k]) + '\t' +
                        str(u_jk[j][k]))
    return u_jk


def read_concept_pair(file_name, c_ij):
    u_jk = []
    with open(file_name, 'r') as f:
        j = -1
        k = -1
        prev_concept = ''
        for line in f:
            pair = line.split('\t')
            if prev_concept != pair[0]:
                j += 1
                k = 0
                u_jk.append(np.zeros(len(c_ij[1]), float))
            k += 1
        u_jk[j][k] = pair[2]


def relevance_wordnet(c_0, c_1):
    # print(str(c_0) + "\t" + str(c_1))
    if type(c_0[0]) == type(()):
        allsyns1 = set(ss for word in c_0 for ss in
                       wn.synsets(word[0], pos=transform_POS(word[1])))
    else:
        allsyns1 = set(ss for ss in wn.synsets(c_0[0],
                                               pos=transform_POS(c_0[1])))
    if type(c_1[0]) == type(()):
        allsyns2 = set(ss for word in c_1 for ss in
                       wn.synsets(word[0], pos=transform_POS(word[1]))
                       if ss not in allsyns1)
    else:
        allsyns2 = set(ss for ss in wn.synsets(c_1[0],
                                               pos=transform_POS(c_1[1])))

    if len(allsyns1) == 0 or len(allsyns2) == 0:
        return 0
    list_sim = [wn.wup_similarity(s1, s2) or 0 for s1, s2 in
                product(allsyns1, allsyns2)]
    best = sum(list_sim)/len(list_sim)
    # print(best)
    return best


def cosine_similarity(c_0, c_1):
    return 0


WN_NOUN = 'n'
WN_VERB = 'v'
WN_ADJECTIVE = 'a'
WN_ADJECTIVE_SATELLITE = 's'
WN_ADVERB = 'r'


def transform_POS(pos):
    pos_dict = \
            {
                'CC': None,
                'CD': None,
                'DT': None,
                'EX': None,
                'FW': None,
                'IN': None,
                'JJ': WN_ADJECTIVE,
                'JJR': WN_ADJECTIVE,
                'JJS': WN_ADJECTIVE,
                'LS': None,
                'MD': None,
                'NN': WN_NOUN,
                'NNS': WN_NOUN,
                'NP': WN_NOUN,
                'NPS': WN_NOUN,
                'PDT': None,
                'POS': None,
                'PP': None,
                'PP$': None,
                'RB': WN_ADVERB,
                'RBR': WN_ADVERB,
                'RBS': WN_ADVERB,
                'RP': None,
                'SYM': None,
                'TO': None,
                'UH': None,
                'VB': WN_VERB,
                'VBD': WN_VERB,
                'VBG': WN_VERB,
                'VBN': WN_VERB,
                'VBP': WN_VERB,
                'VBZ': WN_VERB,
                'WDT': None,
                'WP': None,
                'WP$': None,
                'WRB': None,
                'u': None
            }
    return pos_dict[pos]
