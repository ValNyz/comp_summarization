# -*- coding: utf-8 -*-

"""
load documents which are define in the file topics_{task.name}

__author__ : valentin nyzam
"""

import sys
import os

import preprocess.text as text
import preprocess.util as util

# from globals import root
# from globals import data_root

import logging
logger = logging.getLogger(__name__)


class SummaryProblem:
    """
    a class for representing elements of a summary problem
    self.id               'd0701'
    self.title            'southern poverty law center'
    self.query            <title>: <narr>
    self.docs_a_paths     a list of paths to the input documents
    self.docs_b_paths     a list of paths to the input documents
    self.docs_a           [document1, ... ]
    self.docs_b           [document1, ... ]
    self.annotators       set(['a', 'b', 'c', 'd'])
    self.training         {'a': <summary a>, ... }
    """

    def __init__(self, id, title, docs_a, docs_b):
        self.id = id
        self.title = title
        self.docs_a_paths = docs_a[:]
        self.docs_b_paths = docs_b[:]

        # for checking state
        self.loaded_docs = False
        self.parsed = False
        self.loaded_ir_docs = False

        # variables that might get set later
        self.docs_a = None
        self.docs_b = None
        self.training = {}
        self.annotators = set()

    def load_documents(self):
        self.docs_a = []
        for path in self.docs_a_paths:
            logger.debug(path)
            doc = text.Document(path)
            doc.get_sentences()
            self.docs_a.append(doc)

        self.docs_b = []
        for path in self.docs_b_paths:
            logger.debug(path)
            doc = text.Document(path)
            doc.get_sentences()
            self.docs_b.append(doc)

        self.loaded_docs = True

    def dump_data(self, path, encoding='utf-8'):
        path = os.path.join(path, self.id)
        os.mkdir(path)

        sent_file = '.sent'
        tok_file = '.sent.tok'
        # stem_file = '.sent.tok.stem'
        lemm_file = '.sent.tok.lemm'
        lemm_pos_file = '.sent.tok.lemm.pos'
        # stem_no_stop_file = '.sent.tok.stem.no_stopword'
        doc_file = '.doc'
        par_file = '.par'

        sent_fh = open(os.path.join(path, self.id + "-A" + sent_file), 'w',
                       encoding=encoding)
        tok_fh = open(os.path.join(path, self.id + "-A" + tok_file), 'w',
                       encoding=encoding)
        lemm_fh = open(os.path.join(path, self.id + "-A" + lemm_file), 'w',
                       encoding=encoding)
        # lemm_pos_fh = open(os.path.join(path, self.id + "-A" + lemm_pos_file), 'w',
                       # encoding=encoding)
        doc_fh = open(os.path.join(path, self.id + "-A" + doc_file), 'w',
                       encoding=encoding)
        par_fh = open(os.path.join(path, self.id + "-A" + par_file), 'w',
                       encoding=encoding)
        for doc in self.docs_a:
            # count = 0
            for sent in doc.sentences:
                # cleaning
                if sent.original[0:2].islower():
                    logger.debug('bad parse:', sent.original)
                    continue
                sent_fh.write('%s\n' % sent.original)  # % cleaned)
                tok_fh.write('%s\n' % ' '.join(sent.tokens))
                lemm_fh.write('%s\n' % ' '.join(sent.lemm))
                # lemm_pos_fh.write('%s\n' % ' '.join(sent.lemm_pos))
                doc_fh.write('%s %s\n' % (doc.id, doc.language))
                par_fh.write('%d\n' % int(sent.paragraph_starter))
        sent_fh.close()
        tok_fh.close()
        lemm_fh.close()
        # lemm_pos_fh.close()
        doc_fh.close()
        par_fh.close()
        sent_fh = open(os.path.join(path, self.id + "-B" + sent_file), 'w',
                       encoding=encoding)
        tok_fh = open(os.path.join(path, self.id + "-B" + tok_file), 'w',
                       encoding=encoding)
        lemm_fh = open(os.path.join(path, self.id + "-B" + lemm_file), 'w',
                       encoding=encoding)
        # lemm_pos_fh = open(os.path.join(path, self.id + "-B" + lemm_pos_file), 'w',
                       # encoding=encoding)
        doc_fh = open(os.path.join(path, self.id + "-B" + doc_file), 'w',
                       encoding=encoding)
        par_fh = open(os.path.join(path, self.id + "-B" + par_file), 'w',
                       encoding=encoding)
        for doc in self.docs_b:
            # count = 0
            for sent in doc.sentences:
                # cleaning
                if sent.original[0:2].islower():
                    logger.debug('bad parse:', sent.original.encode(encoding))
                    continue
                sent_fh.write('%s\n' % sent.original)  # % cleaned)
                tok_fh.write('%s\n' % ' '.join(sent.tokens))
                lemm_fh.write('%s\n' % ' '.join(sent.lemm))
                # lemm_pos_fh.write('%s\n' % ' '.join(sent.lemm_pos))
                doc_fh.write('%s %s\n' % (doc.id, doc.language))
                par_fh.write('%d\n' % int(sent.paragraph_starter))
        sent_fh.close()
        tok_fh.close()
        lemm_fh.close()
        # lemm_pos_fh.close()
        doc_fh.close()
        par_fh.close()

    def __str__(self):
        s = []
        s.append('%s summaryproblem' % '#start')
        s.append('id %s' % self.id)
        s.append('title %s' % self.title)
        s.append('narr %s' % self.narr)
        s.append('docs_a %d\n%s' % (len(self.docs_a), '\n'.join(
            ['%s' % n for n in self.docs_a])))
        s.append('docs_b %d\n%s' % (len(self.docs_b), '\n'.join(
            ['%s' % n for n in self.docs_b])))
        for annotator in self.annotators:
            s.append('train %s\n%s' % (annotator, '\n'.join(
                ['%s' % n for n in self.training[annotator]])))
        return '\n'.join(s)


def setup_task(task):
    """
    task.topic_file: xml file for tac
    task.doc_path: path containing source documents
    task.manual_path: path for manual (human) summaries
    """

    # get all document data
    all_docs = {}
    logger.debug(task.doc_path)
    files = util.get_files(task.doc_path,
                                    r'[^_]+_?[^_]*_?\d+[\.\-]\d+')
    logger.debug(files)
    logger.info('Loading [%d] files\n' % len(files))
    for file in files:
        logger.debug(file)
        id = os.path.basename(file)
        all_docs[id] = file

    # initialize problems
    problems = []
    # load XML task definition
    from xml.etree import ElementTree
    root = ElementTree.parse(task.topic_file).getroot()
    for topic in root:
        if topic.tag != "topic":
            continue
        id = topic.attrib["id"]
        title = None
        docsets = []
        docset_ids = []
        for node in topic:
            if node.tag == "title":
                title = node.text.strip()
            elif node.tag == "docsetA":
                documents = node.findall("doc")
                docsets.append([doc.attrib["id"] for doc in documents])
                docset_ids.append(node.attrib["id"])
            elif node.tag == "docsetB":
                documents = node.findall("doc")
                docsets.append([doc.attrib["id"] for doc in documents])
                docset_ids.append(node.attrib["id"])

        if len(docsets) % 2 == 1:
            return
        for docset_index in range(0, len(docsets) - 1, 2):

            # map docids to documents
            docs_A = [all_docs[doc] for doc in docsets[docset_index]]
            docs_B = [all_docs[doc] for doc in docsets[docset_index+1]]

            # create a SummaryProblem
            problem = SummaryProblem(docset_ids[docset_index][:5],
                                     title, docs_A, docs_B)

            problems.append(problem)

    logger.info('Setting up [%d] problems\n' % len(problems))
    task.problems = problems


def setup_sentences(task, parser=None, reload=False, options=None):
    # load problems quickly from pickle file
    if (not reload) and os.path.isfile(task.data_pickle):
        logger.info('Loading [%s] problem data from [%s]\n'
                         % (task.name, task.data_pickle))
        task.problems = util.load_pickle(task.data_pickle)
        return

    # parse sentences
    for problem in task.problems:
        logger.debug('%s\n' % problem.id)
        problem.load_documents()
        if parser:
            for doc in problem.docs_A:
                doc.parse_sentences(parser)
            for doc in problem.docs_B:
                doc.parse_sentences(parser)
            problem.parsed = True

    if parser:
        parser.run()
        for sentence, parsetree in parser.parsed.items():
            sentence.parsed = parsetree

    # save pickled version for faster loading later
    logger.info('Saving [%s] problem data in [%s]\n'
                     % (task.name, task.data_pickle))
    util.save_pickle(task.problems, task.data_pickle)
