# -*- coding: utf-8 -*-

"""
Comparative summarization model using sentence to sentence similarity 

__author__ : Valentin Nyzam
"""

from browse_corpus import build_dicts
from model.comp_model import reuters_idf_dict

import gensim
import os
from time import sleep
import sys

from multiprocessing import Process, Queue, JoinableQueue
# import queue
from globals import THREAD

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)

logger = logging.getLogger(__name__)
fh = logging.FileHandler(__name__ + ".log")
fh.setLevel(logging.INFO)

logger.addHandler(fh)


class Comp_sent_model(object):
    def __init__(self, model_name, l_sents, threshold=0.7):
        self.model_name = model_name
        self.l_sents = l_sents
        self.threshold = threshold
        self.save_name = 'pair_sentence_' + l_sents[0][0].corpus[:-2]
        self.typ = 'sent'
        self.d_sen_sim = {}

        self.prepare()

    def prepare(self):
        self.d_id_word, self.d_word_id = build_dicts(self.l_sents)

        print("Vocab size : " + str(len(self.d_id_word)))

        if os.path.exists(self.save_name + '.model'):
            self._read_sentence_pair(self.save_name + '.model')
        else:
            self._update_model(self.model_name)
            self._make_sentence_pair()
            logger.info('Write sentence pair similarity in ' + self.save_name)
            with open(self.save_name + '.model', 'w', encoding='utf-8') as f:
                for item, value in self.d_sen_sim.items():
                    f.write(str(item[0]) + '\t' + str(item[1]) + '\t' +
                            str(value))
        self.dict_idf = reuters_idf_dict(self.l_sents, 'reuters_idf', 1)

    def _update_model(self, model_name):
        """_update_model
        update word embeddings model on self.l_sents for unseen word
        """
        sents = []
        for doc in self.l_sents:
            sents.extend(doc)
        self.model = gensim.models.Word2Vec.load(model_name)
        self.model.min_count = 0
        self.model.build_vocab(sents, update=True)
        self.model.train(sents, total_examples=len(sents),
                         epochs=self.model.epochs)

    def _read_sentence_pair(self, file_name):
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f:
                pair = line.split('\t')
                s_0 = pair[0]
                s_1 = pair[1]
                if float(pair[2].rstrip()) > self.threshold:
                    self.d_sen_sim[(s_0, s_1)] = pair[2]

    def _make_sentence_pair(self):
        size = len(self.l_sents[0] * len(self.l_sents[1])) 
        logger.info(str(size) + ' concept pairs to test')
        queue_in = JoinableQueue()
        queue_out = Queue()

        producer = Process(target=_pair_producer, args=(self.l_sents[0], self.l_sents[1],
                                                       queue_in))
        producer.start()

        print_thread = Process(target=_print_progress, args=(queue_in, queue_out))
        print_thread.start()

        threads = []
        for _ in range(THREAD):
            t = Process(target=_pair_consumer, args=(queue_in, queue_out,
                                                    self.model, self.l_sents))
            t.start()
            threads.append(t)

        queue_in.join()

        for t in threads:
            t.terminate()

        counter = 0
        while counter < size:
            item = queue_out.get()
            i = item[0]
            j = item[1]
            self.d_sen_sim[(i, j)] = item[2]
            counter += 1

        logger.warning("Processing complete")

        print_thread.terminate()

def _pair_producer(l_sents1, l_sents2, queue):
    for i in range(len(l_sents1)):
        for j in range(len(l_sents2)):
            queue.put((i, j))
            # logger.info('Producer : ' + str(queue.qsize()))
    logger.warning('Queuing complete')

def _pair_consumer(queue_in, queue_out, model, l_sents):
    while True:
        item = queue_in.get()

        i = item[0]
        j = item[1]

        # with lock:
        queue_out.put((i, j, 1./(1.+model.wmdistance(l_sents[0][i],
                                                  l_sents[1][j]))))
        queue_in.task_done()

def _print_progress(queue_in, queue_out):
    counter = 0
    while True:
        if counter < 10:
            counter += 1
        else:
            counter = 0
            logger.warning('Pairing queue : ' + str(queue_in.qsize()))
            logger.warning('Processing queue : ' + str(queue_out.qsize()))
        sleep(0.5)
