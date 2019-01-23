# -*- coding: utf-8 -*-

"""
    text specification :
    stopwords removal
    stemming
    sentence splitting
    tokenizing

__author__ : Valentin Nyzam
"""
import os
import sys
import re
import nltk
import string
# import _stopwords

from nltk import SnowballStemmer
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
# from nltk.tokenize.moses import MosesTokenizer
# from nltk.corpus import stopwords
# from treetagger.treetagger import TreeTagger

import preprocess.util as util
from globals import STOPWORDS


class TextProcessor:
    def __init__(self):
        # self._no_punct_pattern = re.compile('[a-zA-Z0-9- ]')
        self._tok = ToktokTokenizer()
        # self._tok = MosesTokenizer(lang='en')
        self._stemmer = SnowballStemmer('english')
        self._lemmatizer = WordNetLemmatizer() # TreeTagger(language='english')
        self._stopwords = set(open(STOPWORDS).read().splitlines())
        # istopwords.words('french') #
        self._porter_stemmer = nltk.stem.porter.PorterStemmer()
        # self._sent_tokenizer = util.load_pickle('%s%s'
        # % (STATIC_DATA_ROOT, 'punkt/m07_punkt.pickle'))
        # self._sent_split_ABBR_LIST = set(['Mr.', 'Mrs.', 'Sen.', 'No.',
        # 'Dr.', 'Gen.', 'St.', 'Lt.', 'Col.', 'Capt.'])
        # self._sent_split_PUNCT_LIST = set(['\" ', '\")', ') ', '\' ',
        # '\"\''])

    def sent_split(self, text):
        return nltk.sent_tokenize(text, language='english')

    def tokenize(self, text):
        return self._tok.tokenize(text) #, escape=False)

    def porter_stem(self, word):
        return self._porter_stemmer.stem(word)

    def remove_stopwords(self, words):
        return [w for w in words if w not in self._stopwords]

    def remove_pos_stopwords(self, words, pos):
        list_lemm = []
        # list_pos = []
        for w, p in zip(words, pos):
            if w not in self._stopwords:
                list_lemm.append(w)
                # list_pos.append(p)
        return list_lemm #, list_pos

    def is_just_stopwords(self, words):
        if type(words) == type(''):
            words = words.split()
        for word in words:
            if word not in self._stopwords:
                return False
        return True

    def remove_punct(self, sentence):
        """
        Remove punctuation from sentence as str
        :param sentence: str: sentence with punctuation
        :return: str: sentence without punctuation
        """
        return re.sub('[' + string.punctuation + ']+', '', sentence).strip()
        # return re.sub(r'[^a-zA-Z0-9- ]', '', sentence).strip()

    def remove_punct_sent(self, sentence):
        return [self.remove_punct(word) for word in sentence
                if len(self.remove_punct(word)) > 0]

    def is_punct(self, text):
        """
        returns true if the text (str) consists solely of non alpha-numeric
        characters
        """
        for letter in text.lower():
            if letter not in set(string.punctuation):
                return False

    def lemm_sent(self, sen_pos):
        if self._lemmatizer is None:
            return sent
        else:
            lemm_sent = []
            for tup in sen_pos:
                lemm_sent.append(self._lemmatizer.lemmatize(tup[0], get_wordnet_pos(tup[1])))
            return lemm_sent
            # return lemm_sent , lemm_pos

    def stem_sent(self, sent):
        return [self.stem(word) for word in sent]

    def stem(self, word):
        if self._stemmer is None:
            return word
        else:
            return self._stemmer.stem(word)


def transform_POS(pos):
    global WORDNET
    return pos.lower()[0] if WORDNET else pos


text_processor = TextProcessor()


class Sentence:
    """
    class for holding information about a single sentence
    self.original     original text string
    self.parsed       s-exp representation of a parse tree
    """

    def __init__(self, text, order=0, source="?", date="?"):
        self.order = order
        self.date = date
        self.source = source
        self.set_text(text)

    def set_text(self, text):
        self.original = text.strip()
        self.parsed = None
        self.length = len(self.original.split())
        self.tokens = [tok.strip() for tok in
                       text_processor.tokenize(self.original.lower())]
        self.pos = nltk.pos_tag(self.tokens)
        self.stemmed = [text_processor.porter_stem(tok) for tok in self.tokens]
        self.lemm = text_processor.lemm_sent(self.pos)
        self.no_stop = [text_processor.porter_stem(tok) for tok in
                        text_processor.remove_stopwords(self.tokens)]

        self.no_stop_freq = {}
        for word in self.no_stop:
            if word not in self.no_stop_freq:
                self.no_stop_freq[word] = 1
            else:
                self.no_stop_freq[word] += 1

    def parse(self, parser=None):
        if self.parsed:
            return
        if parser:
            parser.add_job(self, self.original)
        else:
            # parser = CommandLineParser()
            self.parsed = parser.parse(self.original)

    def __str__(self):
        return self.original


class Document:
    """
    Class for storing documents.
    doc = Document(<document_path>) will load the document and parse it for
    desired information.

    Public Member Variables:
    self.id             'XIE19980304.0061'
    self.source         'XIE'
    self.date           '19980304.0061'
    self.paragraphs     ['Par 1 text', 'Par 2 text', ... ]
    self.sentences      ['sent 1 text', 'sent 2 text', ... ]
    """
    @staticmethod
    def _parse_clean(path):
        return open(path).read().splitlines()

    @staticmethod
    def _parse_newswire(data):
        data = data.replace('``', '\"').replace('\'\'', '\"').replace('`',
                                                                      '\'')
        data = data.replace('\n', '\t')
        pattern = re.compile(r'<\/?(p|text|doc)>', re.I | re.M)
        # convert <p> and <text> to paragraph breaks

        data = re.sub(pattern, '\t', data)
        pattern = re.compile(r'<[^>]*>.*?<\/[^>]*>', re.M)
        # remove tagged content
        data = re.sub(pattern, '\t', data)
        pattern = re.compile(r'<[^>]*>', re.M)  # remove remaining tags
        data = re.sub(pattern, ' ', data)
        pattern = re.compile(r'\s+', re.M)
        # text = map(lambda x: re.sub(pattern, ' ', x.strip()),
        # filter(lambda x: x != '', re.split(r' *\t *\t *', data)))
        text = [re.sub(pattern, ' ', x.strip()) for x in
                [x for x in re.split(r' *\t *\t *', data) if x != '']]
        return text

    @staticmethod
    def _fix_newswire(par):
        """
        clean up newswire paragraphs
        """
        fixed = par

        # get rid of leaders in newswire text
        fixed = re.sub('^(.{0,35} )?\(\w{2,10}?\) ?(--?|_) ?', '', fixed)
        fixed = re.sub('^([A-Z]{2,}.{0,30}? (--?|_) ){,2}', '', fixed)

        # replace underscore, dash, double-dash with comma
        fixed = fixed.replace(' _ ', ', ')
        fixed = fixed.replace(' - ', ', ')
        fixed = fixed.replace(' -- ', ', ')
        fixed = re.sub('([\w\d])--([\w\d])', '\\1, \\2', fixed)

        # other fixes
        fixed = re.sub('^(_|--?)', '', fixed)
        fixed = re.sub(re.compile(r' ?&AMP; ?', re.I), '&', fixed)
        fixed = re.sub(' ?&\w{2}; ?', ' ', fixed)
        fixed = fixed.replace(' ,', ',')
        fixed = re.sub('^, ', '', fixed)
        fixed = re.sub('\s+', ' ', fixed)
        fixed = re.sub('(\w)\.("?[A-Z])', '\\1. \\2', fixed)
        fixed = fixed.strip()

        if text_processor.is_punct(fixed):
            fixed = ''
        return fixed

    def get_sentences(self):
        self.sentences = []
        order = 0
        for par in self.paragraphs:
            # sents_text = text_processor.split_sents(par)
            # sents_text = text_processor.splitta(par)
            sents_text = text_processor.sent_split(par)
            # sents_text_glued = glue_quotes(sents_text)
            par_sent_count = 0
            for sent_text in sents_text:
                if order == 0 and re.search('By [A-Z]', sent_text):
                    continue
                if order == 0 and sent_text.startswith('('):
                    continue
                if order == 0 and re.search('c\.\d', sent_text):
                    continue
                if order == 0 and sent_text.startswith('"') \
                   and sent_text.endswith('"'):
                    continue
                if sent_text.isupper():
                    continue
                if 1.0 * len([1 for c in sent_text if c.isupper()]) \
                   / len(sent_text) > 0.2:
                    continue
                if len(sent_text.split()) < 20 and not re.search('\.[")]?$',
                                                                 sent_text):
                    continue
                if re.search(re.compile('eds:', re.I), sent_text):
                    continue
                if re.search('[ \-]\d\d\d-\d\d\d\d', sent_text):
                    continue
                if '(k)' in sent_text:
                    continue
                sentence = Sentence(sent_text, order, self.source, self.date)
                if par_sent_count == 0:
                    sentence.paragraph_starter = True
                else:
                    sentence.paragraph_starter = False
                self.sentences.append(sentence)
                order += 1
                par_sent_count += 1
        # print
        # self.id, len(self.sentences)

    def parse_sentences(self, parser=None):
        if parser:
            for sentence in self.sentences:
                sentence.parse(parser)
        else:
            # parser = CommandLineParser(BERKELEY_PARSER_CMD)
            for sentence in self.sentences:
                sentence.parse(parser)
            parser.run()
            for sentence in parser.parsed:
                sentence.parsed = parser.parsed[sentence]

    def __init__(self, path, is_clean=False, encoding='utf-8'):
        """
        path is the location of the file to process
        is_clean=True means that file has no XML or other markup: just text
        """
        self.id = 'NONE'
        self.date = 'NONE'
        self.source = 'NONE'
        self.paragraphs = []
        self._isempty = True

        # get generic info
        if os.path.isfile(path):
            rawdata = open(path, encoding=encoding).read()
        elif path.strip().startswith('<DOC>'):
            rawdata = path
        else:
            sys.stderr.write('ERROR: could not read: %s\n' % path)
            return

        try:
            self.id = util.remove_tags(re.findall('<DOCNO>[^>]+</DOCNO>',
                                                  rawdata[:100])[0])
        except:
            match = re.search('<DOC id=\"([^"]+)\"', rawdata[:100])
            if match:
                self.id = str(match.groups(1)[0])
            else:
                sys.stderr.write('ERROR: no <DOCNO>/<DOC id=...> tag: %s\n'
                                 % path)

        # source and date from id (assumes newswire style)
        if self.id != 'NONE':
            self.source = re.findall('^[^_\d]*', self.id)[0]
            self.date = self.id.replace(self.source, '')

        # parse various types of newswire xml
        if is_clean:
            text = self._parse_clean(rawdata)
        else:
            text = self._parse_newswire(rawdata)

        if len(text) == 0:
            sys.stderr.write('WARNING: no text read for: %s\n' % path)
            return

        self.paragraphs = []
        for paragraph in text:
            fixed_par = self._fix_newswire(paragraph)
            if fixed_par == '':
                continue
            self.paragraphs.append(fixed_par)

        self._isempty = False

    def __str__(self):
        s = []
        s.append('%s DOCUMENT' % '#START')
        s.append('ID %s' % self.id)
        s.append('SOURCE %s' % self.source)
        s.append('DATE %s' % self.date)
        s.append('TEXT')
        s.extend(self.paragraphs)
        return '\n'.join(s)


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
