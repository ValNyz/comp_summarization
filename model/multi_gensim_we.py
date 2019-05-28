# -*- coding: utf-8 -*-

"""
Loading multilingual word embeddings
__author__ : Valentin Nyzam
"""
import gensim.models.fasttext as fasttext
from gensim.models import translation_matrix

if __name__ == '__main__':
    en_path = '/home/valnyz/PhD/en_cbow_model_4_6'
    fr_path = '/home/valnyz/PhD/fr_cbow_model_4_6'
    pair_file = '/home/valnyz/python/MUSE/data/crosslingual/dictionaries/fr-en.txt'


    en_model = fasttext.load_facebook_vectors(en_path + '.bin')
    # en_model = fasttext.load_facebook_model(en_path + '.bin')
    en_model.vectors = en_model.wv

    fr_model = fasttext.load_facebook_vectors(fr_path + '.bin')
    # fr_model = fasttext.load_facebook_model(fr_path + '.bin')
    fr_model.vectors = fr_model.wv

    word_pair = []
    with open(pair_file, 'r') as f:
        for line in f:
            tup = tuple(line.strip().split())
            if tup[0] in fr_model and tup[1] in en_model:
                word_pair.append(tup)

    transmat = translation_matrix.TranslationMatrix(en_model, fr_model)
    transmat.train(word_pair)
    print('the shape of translation matrix is:',
          transmat.translation_matrix.shape)

    # translation the word
    words = [('one', 'un'), ('two', 'deux'), ('three', 'trois'), ('four',
                                                                  'quatre'), ('five', 'cinq')]
    source_word, target_word = zip(*words)
    translated_word = transmat.translate(source_word, 5)
    print(translated_word)



