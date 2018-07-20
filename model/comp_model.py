# -*- coding: utf-8 -*-

"""
Comparative summarization method using ILP
Comparative News Summarization Using Linear Programming (Huang et al, 2011)
__author__ : Valentin Nyzam
"""

from time import sleep
import os

# import queue
# import threading
from multiprocessing import Process, Queue, JoinableQueue
import queue
from nltk.corpus import reuters
import math
import gensim.models
from itertools import product
from collections import Counter
from nltk.util import ngrams
from nltk.corpus import wordnet as wn
from scipy.spatial.distance import cosine
from model.wmd import word_mover_distance
from globals import THREAD

import logging
logger = logging.getLogger(__name__)


class Comp_model(object):
    def __init__(self, l_sents, threshold=0.4):
        self.l_sents = l_sents
        self.threshold = threshold
        self.save_name = None
        self.typ = 'word'
        self.pair_word = {}
        self.d_concept = {}
        self.d_sentence = {}
        self.c_ij = []
        self.s_ik = []
        self.l_ik = []
        self.w_ij = []
        self.u_jk = {}  # weight for pair jk
        self.s_jk = {}  # sim for pair jk

    def prepare(self):
        self._make_concept()
        print("Vocab size : " + str(len(self.c_ij[0]) + len(self.c_ij[1])))
        print("Nb concept in doc 0 : " + str(len(self.c_ij[0])))
        print("Nb concept in doc 1 : " + str(len(self.c_ij[1])))

        if os.path.exists(self.save_name + '.model'):
            self.u_jk = self._read_concept_pair(self.save_name + '.model')
        else:
            self._make_concept_pair(self.threshold)
            logger.info('Write concept pair similarity in ' + self.save_name)
            with open(self.save_name + '.model', 'w', encoding='utf-8') as f:
                for tup in self.u_jk.keys():
                    j = tup[0]
                    k = tup[1]
                    f.write(' '.join(self.c_ij[0][j]) + '\t' +
                            ' '.join(self.c_ij[1][k]) + '\t' +
                            str(self.u_jk[tup]) + '\t' +
                            str(self.s_jk[tup]) + '\n')

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
                self.d_sentence[id(sent)] = (i, len(self.s_ik))
                self.s_ik[i].append(temp)
                self.l_ik[i].append(len(sentence))
                self.c_ij[i].extend(temp)
            self.w_ij.append(Counter(self.c_ij[i]))
            self.c_ij[i] = list(set(self.c_ij[i]))
        for j, c_0 in enumerate(self.c_ij[0]):
            self.d_concept[c_0] = (j, None)
        for j, c_1 in enumerate(self.c_ij[1]):
            if c_1 in self.d_concept:
                self.d_concept[c_1] = (self.d_concept[c_1][0], j)
                # logger.info("COMP!!!!")
                # print("FOUND")
            else:
                self.d_concept[c_1] = (None, j)

    def _read_concept_pair(self, file_name):
        u_jk = {}
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f:
                pair = line.split('\t')
                c_0 = pair[0]
                c_1 = pair[1]
                if tuple(c_0.split()) not in self.d_concept:
                    continue
                if tuple(c_1.split()) not in self.d_concept:
                    continue
                j = self.d_concept[tuple(c_0.split())][0]
                k = self.d_concept[tuple(c_1.split())][1]
                if float(pair[3].rstrip()) > self.threshold:
                    u_jk[(j, k)] = float(pair[2].rstrip())
        return u_jk

    def _make_concept_pair(self):
        """make_dict_concept_pair
        :param c_ij:
        :param w_ij:
        :param similarity: similarity function: input(concept, concept):
            ouput(0<int<1)
        :return list[np.array]: size [j, k]
        """
        q = JoinableQueue()
        done_q = Queue()

        # manager = Manager()
        # pair_word = manager.dict()

        threads = []
        for i in range(THREAD):
            t = Process(target=self.eval,
                        args=(self.model, self.w_ij, q, done_q,
                              self.sim, self.threshold))
            t.start()
            threads.append(t)

        t = Process(target=_print_progress, args=(q, done_q))
        t.start()
        threads.append(t)
        logger.info(str(len(self.c_ij[0])*len(self.c_ij[1]))
                    + ' concept pairs to test')
        for j, c_0 in enumerate(self.c_ij[0]):
            for k, c_1 in enumerate(self.c_ij[1]):
                tup = (
                    (j, c_0),
                    (k, c_1)
                )
                q.put(tup)
        logger.info('Queuing complete')
        # block until all pair are processed.
        q.join()

        counter_dq = 0
        # while counter_dq < len(self.u_jk):
        while not q.empty() or not done_q.empty():
            try:
                tup = done_q.get()
                j = tup[0][0]
                k = tup[0][1]

                self.u_jk[(j, k)] = tup[1]
                self.s_jk[(j, k)] = tup[2]

                counter_dq += 1
                if counter_dq % 1000 == 0:
                    logger.info(str(counter_dq) + " pair already processed.")
                done_q.task_done()
            except queue.Empty:
                pass
        logger.info("Processing complete")

        # stop workers
        for t in threads:
            t.terminate()

        logger.info('Verify nb pair : ')
        logger.info('Nb pair : ' + str(counter_dq))
        logger.info('Nb pair : ' + str(len(self.u_jk)))


class Comp_wordnet(Comp_model):
    def __init__(self, l_sents):
        Comp_model.__init__(self, l_sents)
        self.save_name = 'pair_wordnet_' + l_sents[0][0].corpus[:-2] + '_'
        self.eval = _evaluate_pair_word
        self.sim = _relevance_wordnet
        self.model = None


def _relevance_wordnet(model, w_0, w_1):
    allsyns1 = set(ss for ss in wn.synsets(w_0))
    allsyns2 = set(ss for ss in wn.synsets(w_1) if ss not in allsyns1)

    if len(allsyns1) == 0 or len(allsyns2) == 0:
        return 0
    list_sim = [wn.wup_similarity(s1, s2) or 0 for s1, s2 in
                product(allsyns1, allsyns2)]
    best = sum(list_sim)/len(list_sim)
    return best


class Comp_we(Comp_model):
    def __init__(self, model_name, l_sents):
        Comp_model.__init__(self, l_sents)
        self.typ = 'word'
        self.save_name = 'pair_' + l_sents[0][0].corpus[:-2] + '_'
        self.eval = _evaluate_pair_word
        self.sim = cosine_similarity
        self._update_model(model_name)

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


def _print_progress(q, done_q):
    counter = 0
    while True:
        if counter < 10:
            counter += 1
        else:
            counter = 0
            logger.info('Pairing queue : ' + str(q.qsize()))
            logger.info('Processing queue : ' + str(done_q.qsize()))
        logger.debug(q.qsize())
        sleep(1)


def _evaluate_pair_word(model, w_ij, q, done_q, similarity, threshold):
    """
    item = ((j, c_0), (k, c_1))
    """
    while True:
        item = q.get()
        c_0 = item[0][1]
        c_1 = item[1][1]
        j = item[0][0]
        k = item[1][0]
        if not isinstance(c_0, tuple):
            c_0 = (c_0, )
        if not isinstance(c_1, tuple):
            c_1 = (c_1, )
        sim = 0
        for w_0 in c_0:
            for w_1 in c_1:
                cos = similarity(model, w_0, w_1)
                sim += cos
        sim = sim / (len(c_0) * len(c_1))
        done_q.put(((j, k), (w_ij[0][c_0]+w_ij[1][c_1])/2, sim))
        q.task_done()


def _evaluate_pair_concept(model, w_ij, q, done_q, similarity, threshold):
    """
    item = ((j, c_0), (k, c_1))
    """
    while True:
        item = q.get()
        c_0 = item[0][1]
        c_1 = item[1][1]
        j = item[0][0]
        k = item[1][0]
        if not isinstance(c_0, tuple):
            c_0 = (c_0, )
        if not isinstance(c_1, tuple):
            c_1 = (c_1, )
        sim = similarity(model, c_0, c_1)
        done_q.put(((j, k), (w_ij[0][c_0]+w_ij[1][c_1])/2, sim))
        q.task_done()


class Comp_we_wmd(Comp_we):
    def __init__(self, model_name, l_sents):
        Comp_model.__init__(self, l_sents)
        self.save_name = 'pair_we_wmd' + l_sents[0][0].corpus[:-2] + '_'
        self.eval = _evaluate_pair_concept
        self.sim = word_mover_distance
        self._update_model(model_name)


class Comp_cluster(Comp_we):
    def __init__(self, model_name, l_sents, k):
        Comp_we.__init__(self, model_name, l_sents)
        self.save_name = 'pair_cluster' + l_sents[0][0].corpus[:-2] + '_'
        self.sents = []
        for sentence in l_sents:
            self.sents.append(sentence.get_list_word())
        self.k = k

    def prepare(self):
        Comp_model.prepare(self)
        threshold = 0.2
        if os.path.exists(self.save_name + str(threshold) + '.model'):
            self.u_jk = self._read_concept_pair(self.save_name +
                                                str(threshold) + '.model')

        else:
            from kmeans import kmeans

            self._clusters = kmeans(self.sents, self.k, 0.001, self.model)
            self._make_concept_pair()
            with open(self.save_name + str(threshold) + '.model', 'w') as f:
                for tup in self.u_jk.keys():
                    j = tup[0]
                    k = tup[1]
                    f.write(' '.join(self.c_ij[0][j]) + '\t' +
                            ' '.join(self.c_ij[1][k]) + '\t' +
                            str(self.u_jk[tup]) + '\n')

    def _make_concept_pair(self):
        q = JoinableQueue()
        done_q = Queue()

        threads = []
        for i in range(THREAD):
            t = Process(target=_evaluate_pair_word,
                        args=(self.model, self.w_ij, q, done_q, self.sim,
                              self.threshold))
            t.start()
            threads.append(t)

        t = Process(target=_print_progress, args=(q, ))
        t.start()
        threads.append(t)

        for cluster in self._clusters:
            lc_0 = set()
            lc_1 = set()
            for sen in cluster:
                tup = self.d_sentence(id(sen))
                if tup[0] == 0:
                    lc_0.update(self.s_ik[tup[0]][tup[1]])
                else:
                    lc_1.update(self.s_ik[tup[0]][tup[1]])
            for c_0 in lc_0:
                for c_1 in lc_1:
                    tup = (
                        (self.d_concept[c_0][1], c_0),
                        (self.d_concept[c_1][1], c_1)
                    )
                    q.put(tup)

        logger.info('Queuing complete')
        while not q.empty() or not done_q.empty():
            try:
                tup = done_q.get()
                j = tup[0][0]
                k = tup[0][1]
                self.u_jk[(j, k)] = tup[1]
            except queue.Empty:
                pass
        # block until all tasks are done
        q.join()
        # stop workers
        for t in threads:
            t.terminate()

        print('Nb pair : ' + str(len(self.u_jk)))


def l_sent_to_str(l_sents):
    return [sent.get_list_word() for sent in l_sents]


def cosine_similarity(model, w1, w2):
    return 1-cosine(model[w1], model[w2])


def make_concept_idf_dict(docs, dict_idf={}, nb=0, order=2):
    """
    generate dictionary of inverse document frequency for word in l_sents
    :param l_sents: list[list[list[str]]] : list of docs as list of sentences
    as list of word
    :return: dict{int : int} : dictionnary of id_word and inverse document
    frequency
    """
    if len(dict_idf) != 0 and nb != 0:
        for concept in dict_idf.keys():
            dict_idf[concept] = nb/math.exp(dict_idf[concept])
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
        dict_idf[concept] = math.log2(float(nb + len(docs))/dict_idf[concept])
    return dict_idf


def reuters_idf_dict(current_docs, file_name, order=2):
    """
    """
    idf_file = file_name + ".idf"
    dict_idf = {}
    if os.path.exists(idf_file):
        with open(idf_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split("\t")
                # print(values)
                dict_idf[tuple(values[0].split())] = float(values[1])
        dict_idf = make_concept_idf_dict(current_docs, dict_idf,
                                         len(reuters.fileids()))
        return dict_idf
    else:
        logger.info("Process reuters idf.")
        l_docs = []
        for fileid in reuters.fileids():
            l_docs.append(reuters.sents(fileids=[fileid]))
        dict_idf = make_concept_idf_dict(l_docs, order=order)
        with open(idf_file, 'w', encoding='utf-8') as f:
            for concept in dict_idf.keys():
                f.write(' '.join(concept) + '\t' + str(dict_idf[concept]) +
                        '\n')
        dict_idf = make_concept_idf_dict(current_docs, dict_idf,
                                         len(reuters.fileids()))
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
