# -*- coding: utf-8 -*-

"""
Comparative summarization method using ILP
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


def score_sentence_ilp(*l_sents):
    ilp = ilp_wordnet(l_sents)
    ilp.prepare()
    generate_ilp_problem(ilp.c_ij, ilp.tf_i, ilp.u_jk, ilp.s_ik, ilp.l_ik)
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
    prob = pulp.lpproblem('summarize', pulp.lpmaximize)

    # oc_ij
    oc_ij = pulp.lpvariable.dicts('concepts', [(i, j) for i in range(len(c_ij))
                                               for j in range(len(c_ij[i]))],
                                  0, 1, pulp.lpbinary)
    # op_jk
    op_jk = pulp.lpvariable.dicts('pairs', [(j, k) for j in range(len(u_jk))
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
                         for k in range(len(c_ij[1]))]) + (1-lambd)*lpSum(
                             [w_ij[i][c_ij[i][j]]*oc_ij[(i, j)] for i in
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


class ilp(object):
    def __init__(self, *l_sents):
        self.l_sents = l_sents
        self.c_ij = []
        self.s_ik = []
        self.l_ik = []
        self.w_ij = []

    def prepare(self):
        self._make_concept()
        print("Vocab size : " + str(len(self.c_ij[0]) + len(self.c_ij[1])))

    def _make_concept(self):
        """_make_concept
        :param *l_sents:
        :param order:
        :return list[list[concept]]: list of all concepts per doc
        :return list[list[list[concept]]]: list of doc as list of sentence as
        list of concepts
        :return list[dict{concept:tf}]
        """
        for i, doc in enumerate(self.l_sents):
            self.c_ij.append([])
            self.s_ik.append(set())
            self.l_ik.append([])
            for sentence in doc:
                sent = sentence.get_list_word_pos()
                temp = list(ngrams(sent, order))
                temp.extend(sent)
                self.s_ik[i].update(temp)
                self.l_ik[i].append(len(sent))
                self.c_ij[i].extend(temp)
            self.w_ij.append(Counter(self.c_ij[i]))
            self.c_ij[i] = set(self.c_ij[i])
        return self.c_ij, self.s_ik, self.l_ik, self.w_ij


class ilp_wordnet(ilp):
    def prepare(self):
        ilp.prepare()
        threshold = 0.2
        if os.path.exists():
            self.u_jk = read_concept_pair('pair_ilp_wordnet_' + str(threshold))
        else:
            self.u_jk = self._make_concept_pair(self.c_ij, self.w_ij)

    def _make_concept_pair(self, threshold):
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
        for j, c_0 in enumerate(self.c_ij[0]):
            u_jk.append(np.zeros(len(self.c_ij[1]), float))
            for k, c_1 in enumerate(self.c_ij[1]):
                if self._relevance_wordnet(c_0, c_1) > threshold:
                    u_jk[j][k] = (self.w_ij[0][c_0]+self.w_ij[1][c_1])/2
                else:
                    u_jk[j][k] = 0
        with open('pair_ilp_wordnet_' + str(threshold),
                  'w') as f:
            for j in range(len(u_jk)):
                for k in range(len(u_jk[j])):
                    f.write(str(self.c_ij[0][j]) + '\t' + str(self.c_ij[1][k])
                            + '\t' + str(u_jk[j][k]))
        return u_jk

    def _relevance_wordnet(cls, c_0, c_1):
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


def ilp_we(ilp):
    def __init__(self, model_name, *l_sents):
        ilp.__init_(l_sents)
        self._update_model(model_name)

    def prepare(self):
        self._update_model()
        self._make_concept()
        print("Vocab size : " + str(len(self.c_ij[0]) + len(self.c_ij[1])))
        threshold = 0.2
        if os.path.exists():
            self.u_jk = read_concept_pair('pair_ilp_wordembdeddings_' +
                                          str(threshold))
        else:
            self.u_jk = self._make_concept_pair(self.c_ij, self.w_ij)

    def _update_model(self, model_name):
        """_update_model
        update word embeddings model on self.l_sents for unseen word
        """
        self.model = gensim.models.Word2Vec.load(model_name)
        self.model.train(chain(self.l_sents))
        return model


    def _make_concept_pair(self):
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
        for j, c_0 in enumerate(self.c_ij[0]):
            u_jk.append(np.zeros(len(self.c_ij[1]), float))
            for k, c_1 in enumerate(self.c_ij[1]):
                sim = self._cosine_similarity(c_0, c_1)
                if sim > threshold:
                    u_jk[j][k] = sim
                    # (self.w_ij[0][c_0]+self.w_ij[1][c_1])/2
                else:
                    u_jk[j][k] = 0
        with open('pair_ilp_we_' + str(threshold), 'w') as f:
            for j in range(len(u_jk)):
                for k in range(len(u_jk[j])):
                    f.write(str(self.c_ij[0][j]) + '\t' + str(self.c_ij[1][k])
                            + '\t' + str(u_jk[j][k]))
        return u_jk


    def _cosine_similarity(self, c_0, c_1):
        if type(c_0[0]) != type(()):
            c_0 = (c_0)
        if type(c_1[0]) != type(()):
            c_1 = (c_1)

        list_sim = [cosine_similarity(self.model, w1, w2) for w1 in t1 for t1 in c_0 for w2
                   in t2 for t2 in c_1]
        best = sum(list_sim)/len(list_sim)
        return best


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


def cosine_similarity(model, w1, w2):
    return 1-cosine(model[w1], model[w2])


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
