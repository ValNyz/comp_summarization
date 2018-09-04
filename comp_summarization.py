# -*- coding: utf-8 -*-

"""
    package for launching comparative summarization of text preprocessed with
    preprocess package
__author__ : Valentin Nyzam
"""

import time
import os
import browse_corpus
from model import comp_model
from globals import WE_MODEL
from knapsack import score_sentence_knapsack
from ilp import score_sentence_ilp
from minimum_dominating_set import score_sentences_MDS
# from cross_entropy_summary import score_sentences_cross_entropy
# from k_core_summary import score_sentences_k_core
# from globals import ROOT as ROOT
# from globals import DATA_ROOT as DATA_ROOT

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def score_sentence_by_word(sent, dict):
    list = [dict[word] for word in sent]
    return 0 if len(list) == 0 else sum(list)/len(list)


def make_comp_summary(comparative, model, threshold,
                      data_path, corpus_id, summ_path, length, options):
    # load sents
    s_A, sents_A = browse_corpus.load_sents(data_path, corpus_id + '-A')
    s_B, sents_B = browse_corpus.load_sents(data_path, corpus_id + '-B')

    start = time.process_time()
    c_model = None
    summary_A, summary_B = None, None
    l_sents = [sents_A, sents_B]
    if model == 'WordNet':
        c_model = comp_model.Comp_wordnet(l_sents, threshold)
    elif model == 'WE':
        c_model = comp_model.Comp_we(WE_MODEL, l_sents, threshold)
    elif model == 'WE_WMD':
        c_model = comp_model.Comp_we_wmd(WE_MODEL, l_sents, threshold)
    
    if comparative == 'knapsack':
        summary_A, summary_B = score_sentence_knapsack(c_model, threshold, sents_A, sents_B)
    elif comparative == 'ilp':
        summary_A, summary_B = score_sentence_ilp(c_model, threshold, sents_A, sents_B)

    # summary_A, summary_B = score_sentences_MDS(sents_A, sents_B)
    # dict_sent_A, dict_sent_B = score_sentences_k_core(sents_A, sents_B, \
    #       score_sentence_by_word)
    # select_n_sentence(5, dict_sent_A, dict_sent_B, s_A, s_B)
    end = time.process_time()
    print("Time elapsed : " + str(end - start))
    return summary_A, summary_B


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

    parser.add_option('-t', '--threshold', dest='threshold', type='float',
                      help='threshold for selection of comparative pair')
    parser.add_option('-k', '--task', dest='task', type='str',
                      help='task name')
    parser.add_option('--docpath', dest='docpath', type='str',
                      help='source document path')
    parser.add_option('-m', '--model',  dest='model', type='str',
                      help='model type (WordNet, WE, WE_WMD, Cluster)')
    parser.add_option('-c', '--comparative',  dest='comparative', type='str',
                      help='comparative method (ILP, Knapsack)')
    parser.add_option('-l', '--length', dest='length', type='int',
                      help='maximum number of words in summaries', default=100)
    parser.add_option('-i', '--input-path', dest='inputpath', type='str',
                      help='path of input files')
    parser.add_option('-o', '--output-path', dest='outpath', type='str',
                      help='path to store output', default='output')
    parser.add_option('-r', '--rouge_command', dest='rouge_command',
                      type='str', help='rouge_command: tac, best, second')
    parser.add_option('-e', '--encoding', default='utf-8', type='str',
                      help='character encoding of text input')
    return parser.parse_args()

if __name__ == '__main__':
    options, task = parse_options()
    task = options.task

    path = os.path.join('output', task)

    start = time.process_time()

    # os.popen('mkdir -p %s' % os.path.join(options.outpath, 'summary'))
    os.popen('mkdir -p %s' % os.path.join(path, 'summary'))
    # run through all corpus
    for c_id in os.listdir(path):
        if 'summary' in c_id or 'rouge_settings.xml' in c_id or 'Model' in c_id or 'Peer' in c_id:
            continue
        # new output
        print(c_id)

        summary_A, summary_B = make_comp_summary(options.comparative, options.model,
                                                 options.threshold,
                                                 os.path.join(path, c_id), c_id,
                                                 os.path.join(path, 'summary'),
                                                 options.length, options)

        os.makedirs(os.path.join(path, 'summary', options.comparative + '_' +
                               options.model, str(options.threshold)), exist_ok=True)

        with open(os.path.join(path, 'summary', options.comparative + '_' +
                               options.model, str(options.threshold), c_id + "-A.sum"), 'w') as f:
            for sent in summary_A:
                f.write(str(sent) + '\n')
                print(sent)
        print()
        with open(os.path.join(path, 'summary', options.comparative + '_' +
                               options.model, str(options.threshold), c_id + "-B.sum"), 'w') as f:
            for sent in summary_B:
                f.write(str(sent) + '\n')
                print(sent)
    end = time.process_time()
    print('Global execution time : ' + str(end-start))
