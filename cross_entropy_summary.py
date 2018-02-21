from model import LidstoneNgramModel
from model import build_vocabulary
from model import count_ngrams


def score_sentences_cross_entropy(sents_A, sents_B):
    """score_sentences_cross_entropy

    :param sents_A:
    :param sents_B:
    """
    sents = []
    sents.extend(sents_A)
    sents.extend(sents_B)
    vocab = build_vocabulary(1, *sents)

    return process_entropy(vocab, sents_B, sents_A),
    process_entropy(vocab, sents_A, sents_B)


def process_entropy(vocab, sents_A, sents_B):
    counter = count_ngrams(2, vocab, sents_A)
    model = LidstoneNgramModel(1, counter)

    dict = {}
    for i, sent in enumerate(sents_B):
        if len(sent) > 8:
            dict[i] = model.entropy(sent)
    return dict
