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
import math
import gensim.models
from itertools import product
from collections import Counter
from nltk.util import ngrams
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
from model.wmd import wmd
from .multiling import multiling_we_model
from globals import THREAD
import browse_corpus

import logging
logger = logging.getLogger(__name__)

class Comp_model(object):
    def __init__(self, l_sents, dictionary, tfidf_model, threshold=0.7):
        self.l_sents = l_sents
        self.dictionary = dictionary
        self.tfidf_model = tfidf_model
        self.threshold = threshold
        self.save_name = None
        self.typ = 'word'
        self.pair_word = {}
        self.d_concept = {}
        self.d_sentence = {}
        self.c_ij = []  # list[list[concept]] list of all concepts per doc
        self.s_ik = []  # list of sentence as list of concept
        self.l_ik = []  # list of sentence length
        self.w_ij = []  # dict of concept weigth per doc
        self.u_jk = {}  # comparative weight (average of weigth of concept 1j
                        # and 2k) for pair jk
        self.s_jk = {}  # similarity for pair jk
        # self.nBOW = {}  # list of concept weigth = w_ij[0] + w_ij[1]

    def prepare(self):
        self._make_concept()
        logger.info("Vocab size : " + str(len(self.c_ij[0]) + len(self.c_ij[1])))
        logger.info("Nb concept in doc 0 : " + str(len(self.c_ij[0])))
        logger.info("Nb concept in doc 1 : " + str(len(self.c_ij[1])))

        if os.path.exists(self.save_name):
            self.u_jk = self._read_concept_pair(self.save_name)
        else:
            self._make_concept_pair()
            logger.info('Write concept pair similarity in ' + self.save_name)
            with open(self.save_name, 'w', encoding='utf-8') as f:
                for tup in self.s_jk.keys():
                    j = tup[0]
                    k = tup[1]
                    f.write(' '.join([c for c in self.c_ij[0][j] if c is not
                                      None]) + '\t' +
                            ' '.join([c for c in self.c_ij[1][k] if c is not
                                      None]) + '\t' +
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
            for j, sentence in enumerate(doc):
                if self.typ == 'pos':
                    sent = sentence.get_list_word_pos()
                else:
                    sent = sentence.get_list_word()
                    # sent = sentence.get_list_lemm_no_stop()
                temp = []
                if order > 1:
                    temp.extend(list(ngrams(sent, order)))
                for w in sent:
                    temp.append((w,))
                # temp.extend(sent)
                self.d_sentence[(i, j)] = len(self.s_ik[i])
                self.s_ik[i].append(temp)
                self.l_ik[i].append(len(sentence))
                self.c_ij[i].extend(temp)
            self.w_ij.append(Counter(self.c_ij[i]))
            self.c_ij[i] = list(set(self.c_ij[i]))
        # self.nBOW = self.w_ij[0] + self.w_ij[1]
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
        for _ in range(int(THREAD/2)):
            t = Process(target=self.eval,
                        args=(self.model, self.w_ij, q, done_q,
                              self.sim, self.threshold))
            t.start()
            threads.append(t)

        t = Process(target=_print_progress, args=(q, done_q))
        t.start()
        threads.append(t)
        size = len(self.c_ij[0]) * len(self.c_ij[1])
        logger.info(str(size) + ' concept pairs to test')
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
        while counter_dq < size:
        # while not q.empty() or not done_q.empty():
            try:
                tup = done_q.get()
                j = tup[0][0]
                k = tup[0][1]

                self.u_jk[(j, k)] = tup[1]
                self.s_jk[(j, k)] = tup[2]

                counter_dq += 1
                # if counter_dq % 1000 == 0:
                    # logger.info(str(counter_dq) + " pair already processed.")
                # done_q.task_done()
            except queue.Empty:
                logger.info("Done_q is empty")
                sleep(0.1)
                pass
        logger.info("Processing complete")

        # stop workers
        for t in threads:
            t.terminate()

        # done_q.join()

        logger.info('Verify nb pair : ')
        logger.info('Nb pair : ' + str(counter_dq))
        logger.info('Nb pair : ' + str(len(self.u_jk)))


class Comp_wordnet(Comp_model):
    def __init__(self, l_sents, dictionary, tfidf_model, threshold=None):
        try:
            from nltk.corpus import wordnet as wn
        except ImportError:
            raise Exception("WordNet or nltk package not installed. Please \
                            install it.")

        if threshold is None:
            Comp_model.__init__(self, l_sents)
        else:
            Comp_model.__init__(self, l_sents, dictionary, tfidf_model, threshold)
        self.save_name = os.path.join('generated_models', 'pair_wordnet_' +
                                      l_sents[0][0].corpus[:-2] + '.model')
        self.eval = _evaluate_pair_word
        self.sim = _relevance_wordnet
        self.model = None


def _relevance_wordnet(model, w_0, w_1):
    try:
        allsyns1 = set(ss for ss in wn.synsets(w_0))
        allsyns2 = set(ss for ss in wn.synsets(w_1) if ss not in allsyns1)
    except:
        logger.error("Synsets : " + str(w_0))
        logger.error("Synsets : " + str(w_1))
        return 0

    if len(allsyns1) == 0 or len(allsyns2) == 0:
        return 0
    list_sim = [wn.wup_similarity(s1, s2) or 0 for s1, s2 in
                product(allsyns1, allsyns2)]
    best = sum(list_sim)/len(list_sim)
    return best


class Comp_we(Comp_model):
    def __init__(self, models_path, models_convention_name, l_sents, dictionary, tfidf_model, threshold=None):
        Comp_model.__init__(self, l_sents, dictionary, tfidf_model, threshold)
        self.models_path = models_path
        self.models_convetion_name = models_convetion_name

    def prepare(self):
        if not os.path.exists(self.save_name):
            self._load_model()
        super(Comp_model).prepare()


    def _load_model(self):
        """
        Load word embeddings model
        Models may need to have been updated using update_model.py for unseen word
        """
        sents = []
        for doc in self.l_sents:
            sents.extend(doc)

        if os.path.exists(self.models_path):
            self.model = multiling_we_model.MultilingModels(self.models_path,
                                                            self.models_convention_name,
                                                            sents)
        else:
            raise Exception('Models folder %s not found.' % self.models_path)

        # logger.info("Normalizing word2vec vectors...")
        # self.model.init_sims(replace=True)
        logging.getLogger('gensim.models.base_any2vec').setLevel(logging.WARNING)
        logging.getLogger('gensim.models.keyedvectors').setLevel(logging.WARNING)
        logging.getLogger('gensim.corpora.dictionary').setLevel(logging.WARNING)


class Comp_we_cosine(Comp_we):
    def __init__(self, models_path, models_convention_name, l_sents, dictionary, tfidf_model, threshold=None):
        if threshold is None:
            Comp_we.__init__(self, l_sents, dictionary, tfidf_model)
        else:
            Comp_we.__init__(self, l_sents, dictionary, tfidf_model, threshold)

        self.typ = 'word'
        self.save_name = os.path.join('generated_models', 'pair_' +
                                      l_sents[0][0].corpus[:-2] + '.model')
        self.eval = _evaluate_pair_word
        self.sim = cosine_similarity


class Comp_we_min_cosinus(Comp_we):
    def __init__(self, models_path, models_convention_name, l_sents, dictionary, tfidf_model, threshold=None):
        if threshold is None:
            Comp_we.__init__(self, l_sents, dictionary, tfidf_model)
        else:
            Comp_we.__init__(self, l_sents, dictionary, tfidf_model, threshold)
        self.save_name = os.path.join('generated_models', 'pair_we_min_cosinus_' +
                                      l_sents[0][0].corpus[:-2] + '.model')

        self.eval = _evaluate_pair_concept
        self.sim = min_cosine_similarity


class Comp_we_min_euclidean(Comp_we):
    def __init__(self, models_path, models_convention_name, l_sents, dictionary, tfidf_model, threshold=None):
        if threshold is None:
            Comp_we.__init__(self, l_sents, dictionary, tfidf_model)
        else:
            Comp_we.__init__(self, l_sents, dictionary, tfidf_model, threshold)
        self.save_name = os.path.join('generated_models', 'pair_we_min_euclidean_' +
                                      l_sents[0][0].corpus[:-2] + '.model')

        self.eval = _evaluate_pair_concept
        self.sim = min_euclidean_similarity


class Comp_we_wmd(Comp_we):
    def __init__(self, models_path, models_convention_name, l_sents, dictionary, tfidf_model, threshold=None):
        if threshold is None:
            Comp_we.__init__(self, l_sents, dictionary, tfidf_model)
        else:
            Comp_we.__init__(self, l_sents, dictionary, tfidf_model, threshold)
        self.save_name = os.path.join('generated_models', 'pair_we_wmd_' +
                                      l_sents[0][0].corpus[:-2] + '.model')

        self.eval = _evaluate_pair_concept
        self.sim = wmd


def _print_progress(q, done_q):
    counter = 0
    while True:
        if counter < 10:
            counter += 1
        else:
            counter = 0
            logger.info('Pairing queue : ' + str(q.qsize()))
            logger.info('Processing queue : ' + str(done_q.qsize()))
        # logger.debug(q.qsize())
        sleep(1)


def _evaluate_pair_word(model, w_ij, q, done_q, similarity, threshold):
    """
    item = ((j, c_0), (k, c_1))
    """
    while True:
        try:
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
                if w_0 is not None:
                    for w_1 in c_1:
                        if w_1 is not None:
                            cos = similarity(model, w_0, w_1)
                            sim += cos
            sim = sim / (len(c_0) * len(c_1))
            done_q.put(((j, k), (w_ij[0][c_0]+w_ij[1][c_1])/2, sim))
            q.task_done()
        except queue.Empty:
            sleep(0.001)
            pass


def _evaluate_pair_concept(model, w_ij, q, done_q, similarity, threshold):
    """
    item = ((j, c_0), (k, c_1))
    """
    while True:
        try:
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
        except queue.Empty:
            sleep(0.001)
            pass


def l_sent_to_str(l_sents):
    return [sent.get_list_word() for sent in l_sents]


def cosine_similarity(model, word1, word2):
    model1 = model[word1.language]
    model2 = model[word2.language]
    return 1-cosine(model1[word1], model2[word2])

def min_cosine_similarity(model, c1, c2):
    value = []
    for w1 in c1:
        for w2 in c2:
            value.append(cosine_similarity(model, w1, w2))
    return min(value)


def euclidean_similarity(model, word1, word2):
    model1 = model[word1.language]
    model2 = model[word2.language]
    return 1/(1+euclidean(model1[word1], model2[word2]))


def min_euclidean_similarity(model, c1, c2):
    value = []
    for w1 in c1:
        for w2 in c2:
            value.append(euclidean_similarity(model, w1, w2))
    return min(value)


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
            # if order > 1:
            set_doc_concept.update(ngrams(sent, order))
            # set_doc_concept.update(ngrams(sent, 1))
            for w in sent:
                set_doc_concept.add(w)
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
    try:
        from nltk.corpus import reuters
    except ImportError:
        raise Exception("Reuters or nltk package not installed. Please \
                        install it.")
    idf_file = file_name + '_' + str(order) + ".idf"
    dict_idf = {}
    if os.path.exists(os.path.join('generated_models', idf_file)):
        l_docs = browse_corpus.list_corpus_2_list_doc(current_docs)
        with open(os.path.join('generated_models', idf_file), 'r',
                  encoding='utf-8') as f:
            for line in f:
                values = line.split("\t")
                # print(values)
                if order > 1:
                    dict_idf[tuple(values[0].split())] = float(values[1])
                else:
                    dict_idf[values[0]] = float(values[1])
        dict_idf = make_concept_idf_dict(l_docs, dict_idf,
                                         len(reuters.fileids()))
        return dict_idf
    else:
        logger.info("Process reuters idf.")
        l_docs = []
        for fileid in reuters.fileids():
            l_docs.append(reuters.sents(fileids=[fileid]))
        dict_idf = make_concept_idf_dict(l_docs, order=order)
        with open(os.path.join('generated_models', idf_file), 'w',
                  encoding='utf-8') as f:
            for concept in dict_idf.keys():
                if isinstance(concept, tuple):
                    f.write(' '.join([c for c in concept if c is not None]) +
                            '\t' + str(dict_idf[concept]) + '\n')
                else:
                    f.write(concept + '\t' + str(dict_idf[concept]) + '\n')
        l_docs = list_sen_corpus_to_list_sen_doc(current_docs, l_docs)
        dict_idf = make_concept_idf_dict(l_docs, dict_idf,
                                         len(reuters.fileids()))
        return dict_idf
