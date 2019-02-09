# -*- coding: utf-8 -*-

"""
Uility for paraphrase_detection including preprocess and other tool.
"""

import numpy as np
import pickle
import os
import gensim

from preprocess.text import clean_text
from browse_corpus import build_dicts

import logging
logger = logging.getLogger(__name__)


def convert_to_sequence(texts, word_to_index, padding=False, size_limit=10):
    sequences = {}
    for idx, tokens in texts.items():
        if padding:
            sequences[idx] = np.array([word_to_index[token]
                                      for i, token in enumerate(tokens)
                                      if i < size_limit] + [0] *
                                      (max(0, size_limit-len(tokens))),
                                      dtype=np.int32)
        else:
            sequences[idx] = np.array([word_to_index[token]
                                      for token in tokens],
                                      dtype=np.int32)
    return sequences


def preprocess(task_path):
    sents_train, train = loadFile(os.path.join(task_path,
                                  'msr_paraphrase_train.txt'))
    sents_test, test = loadFile(os.path.join(task_path,
                                'msr_paraphrase_test.txt'))
    sents = sents_train
    sents.extend(sents_test)
    for i in range(len(sents_test)):
        test['X'][0] += len(sents_train)
        test['X'][1] += len(sents_train)

    index_to_word, word_to_index = build_dicts([sents])
    pickle.dump([sents, index_to_word, word_to_index,
                 train, test], open('data/paraphrase_detection'))


def generate_dataset(task_path, reverse_train=False, autogen=0):
    # Loading data
    [sents,
     index_to_word,
     word_to_index,
     train,
     test] = pickle.load(open('data/paraphrase_detection'))

    # Updating W2V model
    # TODO FastText model
    w2v_model = gensim.models.Word2Vec.load('/home/valnyz/data/reuters_w2v')
    w2v_model.train(sents, total_examples=len(sents),
                    epochs=w2v_model.epochs)

    # Computing the max_seq_length
    max_seq_length = np.max([len(tokens) for idx, tokens in enumerate(sents)])

    # Transforming list of tokens to sequence of indices
    sequences = convert_to_sequence(sents, word_to_index,
                                    padding=True, size_limit=max_seq_length)

    # Training data
    if reverse_train:
        train_X_A = np.zeros((len(train['X'])*2+autogen, max_seq_length),
                             dtype=np.int32)
        train_X_B = np.zeros((len(train['X'])*2+autogen, max_seq_length),
                             dtype=np.int32)
        for i, (s_A, s_B) in enumerate(train['X']):
            train_X_A[i*2, :] = sequences[s_A]
            train_X_A[i*2+1, :] = sequences[s_B]
            train_X_B[i*2, :] = sequences[s_B]
            train_X_B[i*2+1, :] = sequences[s_A]
        train_Y = np.array([train['Y'][i//2]
                            for i in range(len(train['Y'])*2)]+[0]*autogen,
                           dtype=np.int32)
    else:
        train_X_A = np.zeros((len(train['X'])+[0]*autogen, max_seq_length),
                             dtype=np.int32)
        train_X_B = np.zeros((len(train['X'])+[0]*autogen, max_seq_length),
                             dtype=np.int32)
        for i, (s_A, s_B) in enumerate(train['X']):
            train_X_A[i, :] = sequences[s_A]
            train_X_B[i, :] = sequences[s_B]
        train_Y = np.array(train['Y'], dtype=np.int32)

    # Adding automatically generated negative samples
    # from sentences in positive samples
    left, right = zip(*[tup for tup, cls in zip(train['X'], train['Y'])
                        if cls == 1])
    pos_ids = np.array(list(set(left+right)), dtype=np.int32)
    selected_pos_ids = np.random.choice(pos_ids, size=autogen)
    pairs_train_set = set(train['X'])
    pairs_test_set = set(test['X'])
    all_ids = np.array(range(len(sents)), dtype=np.int32)
    starting_i = len(train['X'])*2 if reverse_train else len(train['X'])
    for i, pos_id in enumerate(selected_pos_ids, start=starting_i):
        while True:
            paired_id = np.random.choice(all_ids)
            # Check it is not in test set too
            if ((pos_id, paired_id) not in pairs_train_set and
               (paired_id, pos_id) not in pairs_train_set and
               (pos_id, paired_id) not in pairs_test_set and
               (paired_id, pos_id) not in pairs_test_set):
                train_X_A[i, :] = sequences[pos_id]
                train_X_B[i, :] = sequences[paired_id]
                break
            else:
                logger.debug("Ignoring randomly generated sample that \
                             already exists")

    # Test Data
    test_X_A = np.zeros((len(test['X']), max_seq_length), dtype=np.int32)
    test_X_B = np.zeros((len(test['X']), max_seq_length), dtype=np.int32)

    for i, (s_A, s_B) in enumerate(test['X']):
        test_X_A[i, :] = sequences[s_A]
        test_X_B[i, :] = sequences[s_B]
    test_Y = np.array(test['Y'], dtype=np.int32)

    return w2v_model, index_to_word, word_to_index, train_X_A, train_X_B, \
        train_Y, test_X_A, test_X_B, test_Y


def loadFile(path):
    # Load file from ''path'' and apply ''clean_text'' from ''preprocess.text''
    mrpc_data = {'X_A': [], 'X_B': [], 'y': []}
    sents = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip().split('\t')
            mrpc_data['X'].append((len(sents), len(sents+1)))
            sents.append[text[3]]
            sents.append[text[4]]
            mrpc_data['y'].append(text[0])

    sents = clean_text(sents[1:])
    mrpc_data['X'] = mrpc_data['X'][1:]
    mrpc_data['y'] = [int(s) for s in mrpc_data['y'][1:]]

    return sents, mrpc_data
