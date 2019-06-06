# -*- coding: utf-8 -*-

"""
Comparative summarization model using sentence to sentence similarity 

__author__ : Valentin Nyzam
"""
import os
from time import sleep
import sys
from collections import Counter
from multiprocessing import Process, Queue, JoinableQueue
from globals import THREAD

from lexrank import LexRank
# from model.lexrank_wmd import LexRank_wmd

# from browse_corpus import build_dicts

from model.wmd import wmd

from .multiling import multiling_we_model

import logging
logger = logging.getLogger(__name__)


class Comp_sent_model(object):
    def __init__(self, models_path, models_convention_name, l_sents,
                 dictionary, tfidf_model, lambd=0.55,
                 lexrank=False, update=False):
        self.models_path = models_path
        self.models_convention_name = models_convention_name
        self.l_sents = l_sents
        self.lambd = lambd
        self.save_name = os.path.join('generated_models', 'pair_sentence_wmd_'
                                      + l_sents[0][0].corpus[:-2] + '.model')
        self.typ = 'sent'

        self.dictionary = dictionary
        self.tfidf_model = tfidf_model

        self.d_sen_sim = {}
        self.lexrank = lexrank
        self.update = update

        if not os.path.exists(self.save_name):
            self._load_model()


    def prepare(self):
        self.d_id_sents = []
        self.d_id_sents_corpus = {}
        for i in range(len(self.l_sents)):
            for j in range(len(self.l_sents[i])):
                self.d_id_sents_corpus[(i, j)] = len(self.d_id_sents)
                self.d_id_sents.append((i, j))

        print("Nb sentence : " + str(len(self.d_id_sents)))

        if os.path.exists(self.save_name):
            self._read_sentence_pair(self.save_name)
        else:
            self._make_sentence_pair()
            logger.info('Write sentence pair similarity in ' + self.save_name)
            with open(self.save_name, 'w', encoding='utf-8') as f:
                for item, value in self.d_sen_sim.items():
                    f.write(str(item[0]) + '\t' + str(item[1]) + '\t' +
                            str(value) + '\n')
        if self.lexrank:
            print('Lexrank')
            # lxr = LexRank()
            # self.d_sen_score = lxr.rank_sentences(*l_sents)
            # lxr = LexRank_wmd()
            # self.d_sen_score = lxr.rank_sentences(len(self.d_id_sents), self.d_sen_sim)
        else:
            self.d_sen_score = []
            self.d_word_tfidf = []
            for corpus in self.l_sents:
                word_tf_idf = dict(self.tfidf_model[self.dictionary.doc2bow([word for sen in corpus
                                                                             for word in sen])])
                self.d_word_tfidf.append(word_tf_idf)
                for sen in corpus:
                    score = 0
                    for word in sen:
                        try:
                            score += word_tf_idf[self.dictionary.token2id[word]]
                        except:
                            pass
                    self.d_sen_score.append(score)
        self.max_score = max(self.d_sen_score)
        self.min_score = min(self.d_sen_score)
        diff = self.max_score - self.min_score
        for i in range(len(self.d_sen_score)):
            self.d_sen_score[i] = (self.d_sen_score[i] - self.min_score) / diff


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

        logging.getLogger('gensim.models.base_any2vec').setLevel(logging.WARNING)
        logging.getLogger('gensim.models.keyedvectors').setLevel(logging.WARNING)
        logging.getLogger('gensim.corpora.dictionary').setLevel(logging.WARNING)

    def _read_sentence_pair(self, file_name):
        logger.info('Reading sentence pair similarity model %s' % file_name)
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f:
                pair = line.split('\t')
                s_0 = int(pair[0])
                s_1 = int(pair[1])
                self.d_sen_sim[(s_0, s_1)] = float(pair[2].strip())
        # temp_sorted = sorted(self.d_sen_sim.items(), reverse=True, key= lambda tup: tup[1])
        # for i in range(90):
            # tup = temp_sorted[i]
            # s1 = self.d_id_sents[tup[0][0]]
            # s2 = self.d_id_sents[tup[0][1]]
            # if s1[0] != s2[0]:
                # print(s1)
                # print(s2)
                # s1 = self.l_sents[s1[0]][s1[1]]
                # s2 = self.l_sents[s2[0]][s2[1]]
                # print(str(s1) + "\n" + str(s2) + "\n" + str(tup[1]))

    def _make_sentence_pair(self):
        nb_sen_1 = len(self.l_sents[0])
        nb_sen_2 = len(self.l_sents[1])
        size = nb_sen_1*nb_sen_2
        logger.info(str(size) + ' concept pairs to test')
        queue_in = JoinableQueue()
        queue_out = Queue()

        # producer = Process(target=_pair_producer, args=(len(self.d_id_sents),
                                                        # queue_in))
        # producer.start()


        threads = []
        for _ in range(int(THREAD/2)):
            t = Process(target=_pair_consumer, args=(queue_in, queue_out,
                                                     self.model,
                                                     self.d_id_sents,
                                                     self.l_sents))
            t.start()
            threads.append(t)

        print_thread = Process(target=_print_progress, args=(queue_in, queue_out))
        print_thread.start()
        threads.append(print_thread)

        for i in range(nb_sen_1):
            for j in range(nb_sen_1, nb_sen_1+nb_sen_2):
                queue_in.put((i, j))
        # for i in range(len(self.d_id_sents)):
            # for j in range(i+1, len(self.d_id_sents)):
                # queue_in.put((i, j))
                # logger.info('Producer : ' + str(queue.qsize()))
        logger.info('Queuing complete')

        sleep(0.2)
        # queue_in.join()

        counter = 0
        # while not (queue_in.empty() and queue_out.empty()):
        while counter < size:
            try:
                item = queue_out.get()
                i = item[0]
                j = item[1]
                self.d_sen_sim[(i, j)] = item[2]
                counter += 1
            except queue.Empty:
                logger.warning('queue_out is empty.')
                sleep(0.2)

        max_sim = max(self.d_sen_sim.values())
        min_sim = min(self.d_sen_sim.values())
        for key in self.d_sen_sim.keys():
            self.d_sen_sim[key] = (self.d_sen_sim[key] - min_sim) / (max_sim - \
                                  min_sim)

        logger.info("Processing complete")

        # stop workers
        for t in threads:
            t.terminate()

# def _pair_producer(nb_sen, queue):
    # for i in range(nb_sen):
        # for j in range(i+1, nb_sen):
            # queue.put((i, j))
    # logger.info('Queuing complete')

def _pair_consumer(queue_in, queue_out, model, d_id_sents, l_sents):
    while True:
        item = queue_in.get()

        i = item[0]
        j = item[1]
        sent1 = l_sents[d_id_sents[i][0]][d_id_sents[i][1]]
        sent2 = l_sents[d_id_sents[j][0]][d_id_sents[j][1]]
        sim = wmd(model, sent1, sent2)
        queue_out.put((i, j, 1./(1.+ sim)))
        queue_in.task_done()

def _print_progress(queue_in, queue_out):
    counter = 0
    while True:
        if counter < 10:
            counter += 1
        else:
            counter = 0
            logger.info('Pairing queue : ' + str(queue_in.qsize()))
            logger.info('Processing queue : ' + str(queue_out.qsize()))
        sleep(0.5)
