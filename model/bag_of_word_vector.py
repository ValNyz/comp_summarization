# import model.counter as counter
import math
from collections import Counter
from itertools import chain
import numpy as np

import logging
logger = logging.getLogger(__name__)


def generate_isf_dict(vocab_word, *l_sents):
    """
    generate dictionary of inverse sentence frequency for word in l_sents
    :param l_sents: list[list[str]] : list of sentences (as list of word)
    :return: dict{int : int} : dictionnary of id_word and inverse sentence
    frequency
    """
    sents = chain(*l_sents)
    n = 0
    dict_isf = {}
    for sent in sents:
        for word in set(sent):
            if vocab_word[word] in dict_isf:
                dict_isf[vocab_word[word]] += 1.
            else:
                dict_isf[vocab_word[word]] = 1.
        n += 1
    for id_word in dict_isf.keys():
        dict_isf[id_word] = math.log2(float(n)/dict_isf[id_word])
    return dict_isf


def generate_tf_sent_dict(vocab_word, vocab_id, *l_docs):
    dict_doc = {}
    docs = chain(l_docs)
    for i, sents in enumerate(docs):
        dict_doc[i] = []
        for j, sent in enumerate(sents):
            dict_doc[i].append(np.zeros(len(vocab_word), int))
            for word in sent:
                # logger.debug(word)
                # logger.debug(vocab_word[word])
                dict_doc[i][j][vocab_word[word]] += 1
    return dict_doc


def generate_tf_dict(vocab_word, vocab_id, *l_docs):
    dict_doc = {}
    docs = chain(*l_docs)
    for i, sents in enumerate(docs):
        dict_doc[i] = np.zeros(len(vocab_word), int)
        for sent in sents:
            for word in sent:
                dict_doc[i][vocab_word[word]] += 1
    return dict_doc


def generate_dict_vect(vocab_word, vocab_id, dict_tf, dict_idf, *l_docs):
    """
    :param vocab_word:
    :param vocab_id:
    :param dict_idf:
    :param l_docs: list[list[list[str]]] : list of doc as list of sentences
    (as list of word)
    :return: dict{int : list(int)} : dict_sent : dict of id_sent:tf_idf vector
    """
    docs = chain(l_docs)
    dict_sent = {}
    for sents in docs:
        for sent in sents:
            dict_sent[id(sent)] = np.zeros(len(vocab_word), float)
            for word in sent:
                dict_sent[id(sent)][vocab_word[word]] += \
                 dict_idf[vocab_word[word]]
    return dict_sent


def generate_sent_dict_vect(vocab_word, vocab_id, dict_sent_tf, dict_idf,
                            *l_sents):
    """
    :param vocab_word:
    :param vocab_id:
    :param dict_idf:
    :param l_docs: list[list[list[str]]] : list of doc as list of sentences
    (as list of word)
    :return: dict{int : list(int)} : dict_sent : dict of id_sent:tf_idf vector
    """
    docs = chain(l_sents)
    dict_sent = {}
    for i, sents in enumerate(docs):
        dict_sent[i] = {}
        for j, sent in enumerate(sents):
            dict_sent[i][j] = np.zeros(len(vocab_word), float)
            for word in sent:
                dict_sent[i][j][vocab_word[word]] += dict_idf[vocab_word[word]]
    return dict_sent


def generate_tf(doc):
    """generate_tf

    :param doc: list[list[str]]: list of sentence as list of word
    """
    return Counter(doc)
