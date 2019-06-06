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


class Token(str):
    def __new__(cls, word, language):
        self = super(Token, cls).__new__(cls, word)
        self.language = language

        return self

class Sentence():
    def __init__(self, corpus, idx, order, orig, doc, language=None, tok=None, lemm=None,
                 parse=None, par=None):
        self.corpus = corpus
        self.id = idx
        self.order = order
        self.orig = orig
        self.doc = doc
        self.language = language
        if tok is not None:
            self.tok = [Token(word, self.language)
                        for word in tp.remove_punct(tok).split()
                        if len(word) > 0]
            self.tok2 = tp.remove_stopwords(self.tok)
        self.len = len(self.tok)
        if lemm is not None:
            # self.pos = pos.split()
            self.lemm = [Token(word, self.language)
                         for word in tp.remove_punct(lemm).split()
                         if len(word) > 0]
            self.lemm2 = tp.remove_stopwords(self.lemm)

        self.parse = parse
        self.new_par = (par == '1')
        self.length = len(self.orig.split())

    def get_list_word(self):
        return self.tok

    def __len__(self):
        return self.len

    def __iter__(self):
        return iter(self.tok)

    def __str__(self):
        return self.orig


def load_sents(input_path, corpus_id, encoding='utf-8'):
    """
    Load sentence from input path for the corpus "corpus_id"
    :param input_path: str: input path for the topic files
    :param corpus_id: str: name of the corpus
    :return: list: list of str: list of word based on get_list_word
    from Sentence
    :return: list: list of Sentence: [Sentence]
    """
    data_path = os.path.join(input_path, corpus_id)
    orig_fh = open(data_path + '.sent', encoding=encoding)
    # tok_fh = open(data_path + '.sent.tok')
    tok_fh = open(data_path + '.sent.tok', encoding=encoding)
    lemm_fh = open(data_path + '.sent.tok.lemm', encoding=encoding)
    doc_fh = open(data_path + '.doc', encoding=encoding)
    par_fh = open(data_path + '.par', encoding=encoding)
    parse_fh = None
    if os.path.exists(data_path + '.sent.tok.parsed'):
        parse_fh = open(data_path + '.sent.tok.parsed', encoding=encoding)

    sents = []
    count = 0
    order = 0
    prev_doc = ''
    while True:
        if parse_fh:
            [corpus, doc, orig, tok, lemm, parse, par] = \
                    [s.strip() for s in [corpus_id, doc_fh.readline(),
                                         orig_fh.readline(),
                                         tok_fh.readline(),
                                         lemm_fh.readline(),
                                         parse_fh.readline(),
                                         par_fh.readline()]]
        else:
            [corpus, doc, orig, tok, lemm, parse, par] = \
                    [s.strip() for s in [corpus_id, doc_fh.readline(),
                                         orig_fh.readline(),
                                         tok_fh.readline(),
                                         lemm_fh.readline(),
                                         "",
                                         par_fh.readline()]]
        if not (doc or orig or tok or lemm or parse):
            break
        info = doc.split()
        doc = info[0]
        language = info[1]
        if doc != prev_doc:
            order = 0
        s = Sentence(corpus, count, order, orig, doc, language, tok, lemm, parse, par)
        if len(s.tok) > 8:
            sents.append(s)
            order += 1
            count += 1
        prev_doc = doc

    print('topic [%s]: got [%d] sentences' % (corpus_id, count))
    return [sen.get_list_word() for sen in sents], sents


def list_corpus_2_list_doc(current_corpus, l_docs=[]):
    """
    Transform a list of corpus as a list of documents using browse_corpus.Sentence.doc
    property

    Parameters
        ----------
        current_corpus : :class: list of list of browse_corpus.Sentence
            list of corpus (as a list of Sentence)
        l_docs : :class: list of list of browse_corpus.Sentence
            list of document (as a list of Sentence)

        Returns
        -------
        l_docs : :class: list of list of browse_corpus.Sentence
            list of document (as a list of Sentence)
    """
    doc = []
    for corpus in current_corpus:
        prev_doc = corpus[0].doc
        for sent in corpus:
            doc.append(sent)
            if sent.doc != prev_doc:
                l_docs.append(doc)
                doc = []
            prev_doc = sent.doc
        l_docs.append(doc)
        doc = []
    return l_docs
