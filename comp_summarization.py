# -*- coding: utf-8 -*-

"""
    package for launching comparative summarization of text preprocessed with
    preprocess package
__author__ : Valentin Nyzam
"""

import time
import os
import browse_corpus
from comp_ilp import score_sentence_ilp
# from minimum_dominating_set import score_sentences_MDS
# from cross_entropy_summary import score_sentences_cross_entropy
# from k_core_summary import score_sentences_k_core
# from globals import ROOT as ROOT
# from globals import DATA_ROOT as DATA_ROOT

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def score_sentence_by_word(sent, dict):
    list = [dict[word] for word in sent]
    return 0 if len(list) == 0 else sum(list)/len(list)


def make_comp_summary(data_path, corpus_id, summ_path, length, options):
    # load sents
    s_A, sents_A = browse_corpus.load_sents(data_path, corpus_id + '-A')
    s_B, sents_B = browse_corpus.load_sents(data_path, corpus_id + '-B')

    score_sentence_ilp(sents_A, sents_B)
    # score_sentences_MDS(sents_A, sents_B)
    # dict_sent_A, dict_sent_B = score_sentences_k_core(sents_A, sents_B, \
    #       score_sentence_by_word)
    # select_n_sentence(5, dict_sent_A, dict_sent_B, s_A, s_B)


def select_n_sentence(n, dict_sent_A, dict_sent_B, list_orig_sen_A,
                      list_orig_sen_B):
    for i, sent in enumerate(reversed(sorted(dict_sent_A,
                                             key=dict_sent_A.get))):
        if i == n:
            break
        print(list_orig_sen_A[sent].orig + "\n" + str(dict_sent_A[sent]))
    print()
    for i, sent in enumerate(reversed(sorted(dict_sent_B,
                                             key=dict_sent_B.get))):
        if i == n:
            break
        print(list_orig_sen_B[sent].orig + "\n" + str(dict_sent_B[sent]))


def parse_options():
    """
    set up command line parser options
    """
    from optparse import OptionParser
    usage = 'usage: %prog [options]'
    parser = OptionParser(usage=usage)

    parser.add_option('-t', '--task', dest='task', type='str',
                      help='task name')
    parser.add_option('--docpath', dest='docpath', type='str',
                      help='source document path')
    parser.add_option('--manpath', dest='manpath', type='str',
                      help='manual summary path')
    parser.add_option('-l', '--length', dest='length', type='int',
                      help='maximum number of words in summaries', default=100)
    parser.add_option('-i', '--input-path', dest='inputpath', type='str',
                      help='path of input files')
    parser.add_option('-o', '--output-path', dest='outpath', type='str',
                      help='path to store output', default='output')
    parser.add_option('-r', '--rouge_command', dest='rouge_command',
                      type='str', help='rouge_command: tac, best, second')
    return parser.parse_args()


if __name__ == '__main__':
    options, task = parse_options()

    setup_start_time = time.time()

    # run through all corpus
    for id in os.listdir(options.inputpath):
        if '-C' in id:
            continue
        # new output
        print(id)
        os.popen('mkdir -p %s' % os.path.join(options.outpath, id, 'summary'))
        make_comp_summary(os.path.join(options.outpath, id), id,
                          os.path.join(options.outpath, id, 'summary'),
                          options.length, options)
