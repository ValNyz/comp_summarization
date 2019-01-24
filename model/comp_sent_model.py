# -*- coding: utf-8 -*-

"""
Comparative summarization model using sentence to sentence similarity 

__author__ : Valentin Nyzam
"""

from browse_corpus import build_dicts
from model.comp_model import reuters_idf_dict
from model.lexrank_wmd import LexRank_wmd

from model.wmd import gensim_wmd
from model.wmd import word_mover_distance

import gensim
import os
from time import sleep
import sys

from multiprocessing import Process, Queue, JoinableQueue
from globals import THREAD

import logging
logger = logging.getLogger(__name__)


class Comp_sent_model(object):
    def __init__(self, model_name, l_sents, threshold=0.7):
        self.model_name = model_name
        self.l_sents = l_sents
        self.threshold = threshold
        self.save_name = 'pair_sentence_wmd_' + l_sents[0][0].corpus[:-2]
        self.typ = 'sent'
        self.d_sen_sim = {}

    def prepare(self):
        self.d_id_sents = []
        self.d_id_sents_corpus = {}
        for i in range(len(self.l_sents)):
            for j in range(len(self.l_sents[i])):
                self.d_id_sents.append((i, j))
                self.d_id_sents_corpus[(i, j)] = len(self.d_id_sents)
        self.docs = []
        doc = []
        for corpus in self.l_sents:
            prev_doc = corpus[0].doc
            for sent in corpus:
                doc.append(sent.lemm2)
                if sent.doc != prev_doc:
                    self.docs.append(doc)
                    doc = []
                prev_doc = sent.doc
            self.docs.append(doc)
            doc = []

        # self.d_id_word, self.d_word_id = build_dicts(self.l_sents)

        print("Nb sentence : " + str(len(self.d_id_sents)))

        if os.path.exists(self.save_name + '.model'):
            self._read_sentence_pair(self.save_name + '.model')
        else:
            self._update_model(self.model_name)
            self._make_sentence_pair()
            logger.info('Write sentence pair similarity in ' + self.save_name)
            with open(self.save_name + '.model', 'w', encoding='utf-8') as f:
                for item, value in self.d_sen_sim.items():
                    f.write(str(item[0]) + '\t' + str(item[1]) + '\t' +
                            str(value) + '\n')
        lxr = LexRank_wmd()

        self.d_sen_score = lxr.rank_sentences(len(self.d_id_sents), self.d_sen_sim)
        logger.info(self.d_sen_score)

    def _update_model(self, model_name):
        """_update_model
        update word embeddings model on self.l_sents for unseen word
        """
        sents = []
        for doc in self.l_sents:
            sents.extend(doc)
        if os.path.exists(model_name):
            self.model = gensim.models.Word2Vec.load(model_name)
            self.model.min_count = 0
            self.model.build_vocab(sents, update=True)
            self.model.train(sents, total_examples=len(sents),
                             epochs=self.model.epochs)
        else:
            self.model = gensim.models.Word2Vec(sents,
                                   size=300,
                                   window=10,
                                   min_count=0,
                                   workers=THREAD)
            self.model.train(sents, total_examples=len(sents), epochs=10)
        logger.info("Normalizing word2vec vectors...")
        self.model.init_sims(replace=True)
        logging.getLogger('gensim.models.base_any2vec').setLevel(logging.WARNING)
        logging.getLogger('gensim.models.keyedvectors').setLevel(logging.WARNING)
        logging.getLogger('gensim.corpora.dictionary').setLevel(logging.WARNING)

    def _read_sentence_pair(self, file_name):
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f:
                pair = line.split('\t')
                s_0 = int(pair[0])
                s_1 = int(pair[1])
                if float(pair[2].strip()) >= self.threshold:
                    self.d_sen_sim[(s_0, s_1)] = float(pair[2].strip())

    def _make_sentence_pair(self):
        size = int(((len(self.d_id_sents)-1)*len(self.d_id_sents))/2)
        logger.info(str(size) + ' concept pairs to test')
        queue_in = JoinableQueue()
        queue_out = Queue()

        # producer = Process(target=_pair_producer, args=(len(self.d_id_sents),
                                                        # queue_in))
        # producer.start()


        threads = []
        for _ in range(THREAD/2):
            t = Process(target=_pair_consumer, args=(queue_in, queue_out,
                                                     self.model,
                                                     self.d_id_sents,
                                                     self.l_sents))
            t.start()
            threads.append(t)

        print_thread = Process(target=_print_progress, args=(queue_in, queue_out))
        print_thread.start()
        threads.append(print_thread)

        for i in range(len(self.d_id_sents)):
            for j in range(i+1, len(self.d_id_sents)):
                queue_in.put((i, j))
                # logger.info('Producer : ' + str(queue.qsize()))
        logger.info('Queuing complete')

        sleep(0.2)
        # queue_in.join()

        counter = 0
        while not (queue_in.empty() and queue_out.empty()):
        # while counter < size:
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
        sim = gensim_wmd(model, sent1, sent2)
        # sim = word_mover_distance(model, sent1, sent2)
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
