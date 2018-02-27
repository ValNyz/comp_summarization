# -*- coding: utf-8 -*-

"""
Comparative summarization method using ILP
Comparative News Summarization Using Linear Programming (Huang et al, 2011)
__author__ : Valentin Nyzam
"""


import os

# import queue
# import threading
from multiprocessing import Process, JoinableQueue, Lock
from nltk.corpus import reuters
import math
import gensim.models
from itertools import product
from collections import Counter
from nltk.util import ngrams
from nltk.corpus import wordnet as wn
import numpy as np
from scipy.spatial.distance import cosine
from model.wmd import word_mover_distance
from globals import THREAD

import logging
logger = logging.getLogger(__name__)


class Comp_model(object):
    def __init__(self, l_sents):
        self.l_sents = l_sents
        self.typ = 'pos'
        self.d_concept = {}
        self.c_ij = []
        self.s_ik = []
        self.l_ik = []
        self.w_ij = []
        self.u_jk = {}

    def prepare(self):
        self._make_concept()
        print("Vocab size : " + str(len(self.c_ij[0]) + len(self.c_ij[1])))
        print("Nb concept in doc 0 : " + str(len(self.c_ij[0])))
        print("Nb concept in doc 1 : " + str(len(self.c_ij[1])))

    def _make_concept(self, order=2):
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
            self.s_ik.append([])
            self.l_ik.append([])
            for sentence in doc:
                if self.typ == 'pos':
                    sent = sentence.get_list_word_pos()
                else:
                    sent = sentence.get_list_word()
                temp = []
                if order > 1:
                    temp.extend(list(ngrams(sent, order)))
                for w in sent:
                    temp.append((w,))
                # temp.extend(sent)
                self.d_concept[id(sent)] = (i, len(self.s_ik))
                self.s_ik[i].append(temp)
                self.l_ik[i].append(len(sent))
                self.c_ij[i].extend(temp)
            self.w_ij.append(Counter(self.c_ij[i]))
            self.c_ij[i] = list(set(self.c_ij[i]))


class Comp_wordnet(Comp_model):
    def prepare(self):
        Comp_model.prepare(self)
        threshold = 0.2
        if os.path.exists():
            self.u_jk = read_concept_pair('pair_wordnet_' + str(threshold))
        else:
            self._make_concept_pair(self.c_ij, self.w_ij)

    def _make_concept_pair(self, threshold):
        """make_dict_concept_pair
        :param c_ij:
        :param w_ij:
        :param similarity: similarity function: input(concept, concept):
            ouput(0<int<1)
        :param threshold:
        :return list[np.array]: size [j, k]
        """
        for c_0 in self.c_ij[0]:
            self.u_jk.append(np.zeros(len(self.c_ij[1]), float))
        with open('pair_wordnet_' + str(threshold),
                  'w') as f:
            for j in range(len(self.u_jk)):
                for k in range(len(self.u_jk[j])):
                    f.write(str(self.c_ij[0][j]) + '\t' + str(self.c_ij[1][k])
                            + '\t' + str(self.u_jk[j][k]))

    def _evaluate_pair(self, u_jk, c_0j, c_1j, w_ij, threshold):
        # build list pair concept
        for j, c_0 in enumerate(self.c_0j):
            for k, c_1 in enumerate(self.c_ij[1]):
                if self._relevance_wordnet(c_0, c_1) > threshold:
                    u_jk[j][k] = (self.w_ij[0][c_0]+self.w_ij[1][c_1])/2
                else:
                    u_jk[j][k] = 0

    def _relevance_wordnet(cls, c_0, c_1):
        # print(str(c_0) + "\t" + str(c_1))
        if not isinstance(c_0[0], tuple):
            allsyns1 = set(ss for word in c_0 for ss in
                           wn.synsets(word[0], pos=transform_POS(word[1])))
        else:
            allsyns1 = set(ss for ss in wn.synsets(c_0[0],
                                                   pos=transform_POS(c_0[1])))
        if not isinstance(c_1[0], tuple):
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


class Comp_we(Comp_model):
    def __init__(self, model_name, l_sents):
        Comp_model.__init__(self, l_sents)
        self.typ = 'word'
        self._update_model(model_name)
        self.lock = Lock()

    def prepare(self):
        Comp_model.prepare(self)
        threshold = 0.2
        if os.path.exists('pair_' + str(threshold)):
            self.u_jk = read_concept_pair('pair_' + str(threshold),
                                          self.c_ij)
        else:
            self._make_concept_pair()

    def _update_model(self, model_name):
        """_update_model
        update word embeddings model on self.l_sents for unseen word
        """
        sents = []
        for ss in self.l_sents:
            sents.extend(ss)
        self.model = gensim.models.Word2Vec.load(model_name)
        self.model.min_count = 0
        self.model.build_vocab(sents, update=True)
        self.model.train(sents, total_examples=len(sents),
                         epochs=self.model.epochs)

    def _make_concept_pair(self, threshold=0.2):
        """_make_dict_concept_pair
        :param c_ij:
        :param w_ij:
        :param similarity: similarity function: input(concept, concept):
            ouput(0<int<1)
        :param threshold:
        :return list[np.array]: size [j, k]
        """
        # build list pair concept
        # for j in range(len(self.c_ij[0])):
        # self.u_jk.append(np.zeros(len(self.c_ij[1]), float))
        q = JoinableQueue()
        threads = []
        for i in range(THREAD):
            t = Process(target=self._evaluate_pair,
                        args=(q, threshold, ))
            t.start()
            threads.append(t)
        for j, c_0 in enumerate(self.c_ij[0]):
            for k, c_1 in enumerate(self.c_ij[1]):
                tup = (
                    (j, c_0),
                    (k, c_1)
                )
                q.put(tup)

        # block until all tasks are done
        q.join()
        # stop workers
        for t in threads:
            t.terminate()
        with open('pair_' + str(threshold), 'w') as f:
            f.write(' '.join([j, k]) + '\t' + str(self.u_jk[(j, k)]) + '\n')

    def _evaluate_pair(self, q, threshold):
        """
        item = ((j, c_0), (k, c_1))
        """
        while True:
            item = q.get()
            c_0 = item[0][1]
            c_1 = item[1][1]
            j = item[0][0]
            k = item[1][0]
            sim = self._similarity(c_0, c_1)
            # print(str(c_0) + '\n' + str(c_1) + '\n' + str(sim))
            if sim > threshold:
                # self.u_jk[j][k] = sim
                self.u_jk[(j, k)] = (self.w_ij[0][c_0]+self.w_ij[1][c_1])/2
            # else:
                # self.u_jk[j][k] = 0
            q.task_done()

    def _similarity(self, c_0, c_1):
        if not isinstance(c_0, tuple):
            c_0 = (c_0, )
        if not isinstance(c_1, tuple):
            c_1 = (c_1, )
        # print(c_0)
        # print(c_1)
        with self.lock:
            list_sim = [cosine_similarity(self.model, w0, w1) for w0 in c_0
                        for w1 in c_1]
        best = sum(list_sim)/len(list_sim)
        # print(best)
        return best


class Comp_we_wmd(Comp_we):
    def _similarity(self, c_0, c_1):
        if not isinstance(c_0, tuple):
            c_0 = (c_0, )
        if not isinstance(c_1, tuple):
            c_1 = (c_1, )
        result = word_mover_distance(c_0, c_1, self.model)
        # print(result)
        return result


class Comp_cluster(Comp_we):
    def __init__(self, model_name, l_sents, k):
        Comp_we.__init__(self, model_name, l_sents)
        self.sents = []
        for sentence in l_sents:
            self.sents.append(sentence.get_list_word())
        self.k = k

    def prepare(self):
        Comp_model.prepare(self)
        threshold = 0.2
        if os.path.exists('pair_cluster' + str(threshold)):
            self.u_jk = read_concept_pair('pair_cluster' + str(threshold),
                                          self.c_ij)
        else:
            from kmeans import kmeans

            self._clusters = kmeans(self.sents, self.k, 0.001, self.model)
            self._make_concept_pair()

    def _make_concept_pair(self):
        lc_0 = set()
        lc_1 = set()
        for sen in self._clusters:
            tup = self.d_concept(id(sen))
            if tup[0] == 0:
                lc_0.update(self.s_ik[tup[0]][tup[1]])
            else:
                lc_1.update(self.s_ik[tup[0]][tup[1]])
        for c_0 in lc_0:
            for c_1 in lc_1:
                pass


def doc_to_str(l_sents):
    return [sent.get_list_word() for sent in l_sents]


def read_concept_pair(file_name, c_ij):
    u_jk = {}
    with open(file_name, 'r') as f:
        # j = -1
        # k = -1
        # prev_concept = ''
        for line in f:
            pair = line.split('\t')
            j = pair[0].split()[0]
            k = pair[0].split()[1]
            # if prev_concept != pair[0]:
            # j += 1
            # k = 0
            # u_jk.append(np.zeros(len(c_ij[1]), float))
            # k += 1
            u_jk[(j, k)] = pair[2]
    return u_jk


def cosine_similarity(model, w1, w2):
    return 1-cosine(model[w1], model[w2])


def make_concept_idf_dict(docs, order=2):
    """
    generate dictionary of inverse document frequency for word in l_sents
    :param l_sents: list[list[list[str]]] : list of docs as list of sentences
    as list of word
    :return: dict{int : int} : dictionnary of id_word and inverse document
    frequency
    """
    dict_idf = {}
    for doc in docs:
        set_doc_concept = set()
        for sent in doc:
            if order > 1:
                set_doc_concept.update(ngrams(sent, order))
            for w in sent:
                set_doc_concept.add((w,))
        for concept in set_doc_concept:
            if concept in dict_idf:
                dict_idf[concept] += 1.
            else:
                dict_idf[concept] = 1.
    # print(dict_idf)
    for concept in dict_idf.keys():
        dict_idf[concept] = math.log2(float(len(docs))/dict_idf[concept])
    return dict_idf


def reuters_idf_dict(current_docs, file_name, order=2):
    """
    """
    idf_file = file_name + ".idf"
    dict_idf = {}
    if os.path.exists(idf_file):
        with open(idf_file, 'r') as f:
            for line in f:
                values = line.split("\t")
                dict_idf[tuple(values[0].split())] = values[1]
        return dict_idf
    else:
        l_docs = []
        for doc in current_docs:
            l_docs.append(doc)
        for fileid in reuters.fileids():
            l_docs.append(reuters.sents(fileids=[fileid]))
        dict_idf = make_concept_idf_dict(l_docs, order)
        with open(idf_file, 'w') as f:
            for concept in dict_idf.keys():
                f.write(' '.join(concept) + '\t' + str(dict_idf[concept]) +
                        '\n')
        return dict_idf


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
