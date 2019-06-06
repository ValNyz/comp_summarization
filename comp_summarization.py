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
from model import comp_sent_model
from globals import THREAD
from globals import LOG_LEVEL
from globals import WE_MODELS_FOLDER
from globals import WE_MODELS_CONVENTION_NAME
from extraction_alg import knapsack, sentence_knapsack
from extraction_alg import ilp, sentence_ilp

from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def make_comp_summary(comparative, model, threshold,
                      corpus_i, length, list_corpus_sen,
                      dictionary, tfidf_model):

    # get sents [[SentsA], [SentsB]]
    l_sents = list_corpus_sen[i]
    # sents_A = list_corpus_sen[corpus_i][0]
    # sents_B = list_corpus_sen[corpus_i][1]

    start = time.process_time()

    if model == 'WordNet':
        c_model = comp_model.Comp_wordnet(l_sents, dictionary,
                                          tfidf_model, threshold)
    elif model == 'WE':
        c_model = comp_model.Comp_we_cosine(WE_MODELS_FOLDER,
                                            WE_MODELS_CONVENTION_NAME,
                                            l_sents, dictionary,
                                            tfidf_model, threshold)
    elif model == 'WE_WMD':
        c_model = comp_model.Comp_we_wmd(WE_MODELS_FOLDER,
                                         WE_MODELS_CONVENTION_NAME,
                                         l_sents, dictionary,
                                         tfidf_model, threshold)
    elif model == 'WE_COS':
        c_model = comp_model.Comp_we_min_cosinus(WE_MODELS_FOLDER,
                                                 WE_MODELS_CONVENTION_NAME,
                                                 l_sents, dictionary,
                                                 tfidf_model, threshold)
    elif model == 'WE_EUC':
        c_model = comp_model.Comp_we_min_euclidean(WE_MODELS_FOLDER,
                                                   WE_MODELS_CONVENTION_NAME,
                                                   l_sents, dictionary,
                                                   tfidf_model, threshold)
    elif model == 'WE_SEN_WMD':
        c_model = comp_sent_model.Comp_sent_model(WE_MODELS_FOLDER,
                                                  WE_MODELS_CONVENTION_NAME,
                                                  l_sents, dictionary,
                                                  tfidf_model,threshold)
    else:
        logger.warning(model + " don't exist.")
        exit()

    c_model.prepare()

    if comparative == 'knapsack':
        summary_A, summary_B = knapsack.score_sentence_knapsack(c_model,
                                                                threshold,
                                                                l_sents)
    elif comparative == 'ilp':
        summary_A, summary_B = ilp.score_sentence_ilp(c_model,
                                                      threshold,
                                                      l_sents)
    elif comparative == 'knapsack2':
        summary_A, summary_B = sentence_knapsack.score_sentence_knapsack2(c_model,
                                                                          threshold,
                                                                          l_sents)
    elif comparative == 'ilp2':
        summary_A, summary_B = sentence_ilp.score_sentence_ilp2(c_model,
                                                                threshold,
                                                                l_sents)
    else:
        logger.warning(model + " don't exist.")
        exit()

    end = time.process_time()
    print("Time elapsed : " + str(end - start))
    return summary_A, summary_B


def parse_options():
    """
    set up command line parser options
    """
    from optparse import OptionParser
    usage = 'usage: %prog [options]'
    parser = OptionParser(usage=usage)

    parser.add_option('-t', '--threshold', dest='threshold', type='float',
                      help='threshold for selection of comparative pair',
                      default=0.55)
    parser.add_option('-k', '--task', dest='task', type='str',
                      help='task name')
    parser.add_option('--docpath', dest='docpath', type='str',
                      help='source document path')
    parser.add_option('-m', '--model',  dest='model', type='str',
                      help='model type (WordNet, WE, WE_WMD, WE_SEN_WMD)')
    parser.add_option('-c', '--comparative',  dest='comparative', type='str',
                      help='comparative method (ilp, knapsack, ilp2, knapsack2)')
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


def __init__():
    import logging
    from logging import StreamHandler
    from logging.handlers import TimedRotatingFileHandler

    # Create the Logger
    logger = logging.getLogger()
    logger.setLevel(LOG_LEVEL)

    if len(logger.handlers) > 0:
        logger.handlers.clear()

    # Create the Handler for logging data to a file
    logger_handler = TimedRotatingFileHandler('logs/comp_summarization.log',
                                              when='D', interval=1,
                                              backupCount=7)
    logger_handler.setLevel(logging.DEBUG)

    # Create the Handler for logging data to console.
    console_handler = StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Create a Formatter for formatting the log messages
    logger_formatter = logging.Formatter('%(name)s - %(threadName)s - %(levelname)s - %(message)s')

    # Add the Formatter to the Handler
    logger_handler.setFormatter(logger_formatter)
    console_handler.setFormatter(logger_formatter)

    # Add the Handler to the Logger
    logger.addHandler(logger_handler)
    logger.addHandler(console_handler)

    return logger

if __name__ == '__main__':
    logger = __init__()
    logger.info('Nb thread : ' + str(THREAD))
    options, task = parse_options()
    task = options.task

    path = os.path.join('output', task)

    start = time.process_time()

    os.makedirs(os.path.join(path, 'summary'), exist_ok=True)

    documents = [] # [Documents : [Sen]]
    list_corpus = []
    list_corpus_sen = []  # [Corpus : [SenA : [], SenB : []]]]]
    # run through all corpus
    for c_id in os.listdir(path):
        if 'summary' in c_id or 'rouge_settings.xml' in c_id or 'Model' in c_id or 'Peer' in c_id:
            continue
        list_corpus.append(c_id)
        # load sents
        s_A, sents_A = browse_corpus.load_sents(os.path.join(path, c_id), c_id + '-A')
        s_B, sents_B = browse_corpus.load_sents(os.path.join(path, c_id), c_id + '-B')
        list_docs = browse_corpus.list_corpus_2_list_doc([sents_A, sents_B])
        # document = []
        # for doc in list_docs:
            # for sen in doc:
                # document.extend(sen.get_list_word())
        # documents.append(document)
        documents.extend([[word for sen in doc for word in sen] for doc in list_docs])
        list_corpus_sen.append([sents_A, sents_B])

    dictionary = Dictionary(documents)
    bow_documents = [dictionary.doc2bow(document) for document in documents]
    tfidf_model = TfidfModel(bow_documents, id2word=dictionary.id2token)

    for i in range(len(list_corpus)):
        c_id = list_corpus[i]
        # new output
        logger.info(c_id)

        summary_A, summary_B = make_comp_summary(options.comparative, options.model,
                                                 options.threshold,
                                                 i,
                                                 options.length,
                                                 list_corpus_sen,
                                                 dictionary,
                                                 tfidf_model)

        os.makedirs(os.path.join(path, 'summary', options.comparative + '_' +
                               options.model, str(options.threshold)), exist_ok=True)

        with open(os.path.join(path, 'summary', options.comparative + '_' +
                               options.model, str(options.threshold), c_id + "-A.sum"), 'w') as f:
            summ = ''
            for sent in summary_A:
                f.write(str(sent) + '\n')
                summ += str(sent) + '\n'
            logger.info(summ)
        with open(os.path.join(path, 'summary', options.comparative + '_' +
                               options.model, str(options.threshold), c_id + "-B.sum"), 'w') as f:
            summ = ''
            for sent in summary_B:
                f.write(str(sent) + '\n')
                summ += str(sent) + '\n'
            logger.info(summ)
    end = time.process_time()
    logger.info('Global execution time : ' + str(end-start))
