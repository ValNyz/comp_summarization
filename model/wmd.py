# -*- coding: utf-8 -*-

"""
Word Mover's Distance using word embeddings
    From Word Embeddings To Document Distances (Kusner et al, 2015)
__author__ : Valentin Nyzam
"""
from gensim.corpora.dictionary import Dictionary
from numpy import double, zeros, sqrt, sum as np_sum

import logging
logger = logging.getLogger(__name__)


def wmd(model, document1, document2):
    return wmd_distance(model[document1.language], model[document2.language],
                      document1, document2)

def wmd_distance(model1, model2, document1, document2):
    """Compute the Word Mover's Distance between two documents.

    Largely inspired by wmd_distance from
    gensim.models.WordEmbeddingsKeyedVectors

    Parameters
    ----------
    model1 : WordEmbeddingsKeyedVectors
        Vectors model for document1
    model2 : WordEmbeddingsKeyedVectors
        Vectors model for document2
    document1 : list of str
        Input document.
    document2 : list of str
        Input document.

    Returns
    -------
    float
        Word Mover's distance between `document1` and `document2`.
    """

    if model1 == model2:
        return model1.wmdistance([word for word in document1], [word for word
                                                                in document2])

    # If pyemd C extension is available, import it.
    # If pyemd is attempted to be used, but isn't installed, ImportError will be raised in wmdistance
    from pyemd import emd

    # Remove out-of-vocabulary words.
    len_pre_oov1 = len(document1)
    len_pre_oov2 = len(document2)
    document1 = [token.word for token in document1 if token in model1]
    document2 = [token.word for token in document2 if token in model2]
    diff1 = len_pre_oov1 - len(document1)
    diff2 = len_pre_oov2 - len(document2)
    # if diff1 > 0 or diff2 > 0:
        # logger.info('Removed %d and %d OOV words from document 1 and 2 (respectively).', diff1, diff2)

    if not document1 or not document2:
        logger.info(
            "At least one of the documents had no words that were in the vocabulary. "
            "Aborting (returning inf)."
        )
        return float('inf')

    dictionary = Dictionary(documents=[document1, document2])
    vocab_len = len(dictionary)

    if vocab_len == 1:
        # Both documents are composed by a single unique token
        return 0.0

    # Sets for faster look-up.
    docset1 = set(document1)
    docset2 = set(document2)

    # Compute distance matrix.
    distance_matrix = zeros((vocab_len, vocab_len), dtype=double)
    for i, t1 in dictionary.items():
        if t1 not in docset1:
            continue

        for j, t2 in dictionary.items():
            if t2 not in docset2 or distance_matrix[i, j] != 0.0:
                continue

            # Compute Euclidean distance between word vectors.
            distance_matrix[i, j] = distance_matrix[j, i] = sqrt(np_sum((model1[t1] - model2[t2])**2))

    if np_sum(distance_matrix) == 0.0:
        # `emd` gets stuck if the distance matrix contains only zeros.
        # logger.info('The distance matrix is all zeros. Aborting (returning inf).')
        return float('inf')

    def nbow(document):
        d = zeros(vocab_len, dtype=double)
        nbow = dictionary.doc2bow(document)  # Word frequencies.
        doc_len = len(document)
        for idx, freq in nbow:
            d[idx] = freq / float(doc_len)  # Normalized word frequencies.
        return d

    # Compute nBOW representation of documents.
    d1 = nbow(document1)
    d2 = nbow(document2)

    # Compute WMD.
    return emd(d1, d2, distance_matrix)

