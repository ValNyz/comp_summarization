# -*- coding: utf-8 -*-

"""
package which permit to work with text documents as
[documents[paragraphs[sentences[words]]]]
__author__ : Valentin Nyzam
"""

# import re
import os
from itertools import chain
from preprocess.text import text_processor as tp


class Sentence:
    def __init__(self, corpus, id, order, orig, doc, tok=None, pos=None,
                 parse=None, par=None, unresolved=False):
        self.corpus = corpus
        self.id = id
        self.order = order
        self.orig = orig
        self.tok = [word for word in tp.remove_punct(tok).split()
                    if len(word) > 0]
        self.pos = pos.split()
        if pos is not None:
            self.tok2, self.lemm_pos = \
                   tp.remove_pos_stopwords(self.tok,
                                           [p for p in
                                            tp.remove_punct(pos).split()
                                            if len(p) > 0])
        else:
            self.tok2 = tp.remove_stopwords(self.tok)
        # self.stem = tp.stem_sent(self.tok2)
        # self.lemm = tp.remove_stopwords([word.lower()
        # for word in tp.lemm_sent(self.orig) if len(word) > 0])
        # print(self.lemm)

        self.doc = doc
        self.parse = parse
        self.new_par = (par == '1')
        self.length = len(self.orig.split())
        self.depends = set()
        self.groups = []
        self.skip = False
        self.skip_concepts = False
        self.unresolved = unresolved
        self.atleast = ""

    def get_list_word(self):
        return self.tok2

    def get_list_word_pos(self):
        return [(w, p) for p in self.lemm_pos for w in self.tok2]

    def __str__(self):
        return self.orig


def load_sents(input_path, corpus_id):
    """
    Load sentence from input path for the corpus "corpus_id"
    :param input_path: str: input path for the topic files
    :param corpus_id: str: name of the corpus
    :return: list: list of str: list of word based on get_list_word
    from Sentence
    :return: list: list of Sentence: [Sentence]
    """
    data_path = os.path.join(input_path, corpus_id)
    orig_fh = open(data_path + '.sent')
    # tok_fh = open(data_path + '.sent.tok')
    tok_fh = open(data_path + '.sent.tok.lemm')
    pos_fh = open(data_path + '.sent.tok.lemm.pos')
    doc_fh = open(data_path + '.doc')
    par_fh = open(data_path + '.par')
    parse_fh = None
    if os.path.exists(data_path + '.sent.tok.parsed'):
        parse_fh = open(data_path + '.sent.tok.parsed')

    sents = []
    count = 0
    order = 0
    prev_doc = ''
    while True:
        if parse_fh:
            [corpus, doc, orig, tok, pos, parse, par] = \
                    [str.strip() for str in [corpus_id, doc_fh.readline(),
                                             orig_fh.readline(),
                                             tok_fh.readline(),
                                             pos_fh.readline(),
                                             parse_fh.readline(),
                                             par_fh.readline()]]
        else:
            [corpus, doc, orig, tok, pos, parse, par] = \
                    [str.strip() for str in [corpus_id, doc_fh.readline(),
                                             orig_fh.readline(),
                                             tok_fh.readline(),
                                             pos_fh.readline(), "",
                                             par_fh.readline()]]
        if not (doc or orig or tok or pos or parse):
            break
        if doc != prev_doc:
            order = 0
        s = Sentence(corpus, count, order, orig, doc, tok, pos, parse, par)
        if len(s.tok2) > 0:
            sents.append(s)
            order += 1
            count += 1
        prev_doc = doc

    print('topic [%s]: got [%d] sentences' % (corpus_id, count))
    return [sen.get_list_word() for sen in sents], sents


def build_dicts(*l_sents):
    """
    build dicts of words for the list of Sentence l_sentences
    :param l_sentences: list: [list of Sentence]
    :return: dict_int: dict of (int:str)
    :return: dict_str: dict of (str:int)
    """
    sents = chain(*l_sents)
    dict_str = {}
    dict_int = {}
    for sen in sents:
        for word in sen:
            if word not in dict_str:
                dict_int[len(dict_int)] = word
                dict_str[word] = len(dict_str)
    return dict_str, dict_int


def str_sent_to_int_sent(dict_str, sent):
    """
    transform a Sentence to a sentence of int based on dict_str
    :param dict_str: dict: dict of (str:int): dict of words
    :param sent: list: list of Sentence
    :return: list: list of int
    """
    return [dict_str(word) for word in sent if word in dict_str]
