# -*- coding: utf_8 -*-

"""
Learn new W2V model using gensim
"""

from nltk.corpus import reuters
import gensim

from preprocess.text import clean_text

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def update_w2v_model(model, new_sents):
    model.train(clean_text(new_sents))


def learn_reuters_w2v_model(path):
    logger.info('*** Loading new files from reuters corpus. ***')
    logger.info('Reuter corpus contains {} news documents'.format(
                len(reuters.fileids())))
    sents = reuters.sents()
    logger.info('Reuter corpus contains {} sentences'.format(
                len(sents)))
    model = gensim.models.Word2Vec(sents,
                                   min_count=1)
    model.save(path)


# learn_reuters_w2v_model('/home/valnyz/data/reuters_w2v')
