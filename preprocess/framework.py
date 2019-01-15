# -*- coding: utf-8 -*-

"""
Load documents which are define in the file topics_{task.name}

__author__ : Valentin Nyzam
"""

import sys
import os
import logging

import preprocess.text as text
import preprocess.util as util

# from globals import ROOT
# from globals import DATA_ROOT

logger = logging.getLogger(__name__)


class SummaryProblem:
    """
    A class for representing elements of a summary problem
    self.id               'D0701'
    self.title            'Southern Poverty Law Center'
    self.query            <title>: <narr>
    self.docs_A_paths     a list of paths to the input documents
    self.docs_B_paths     a list of paths to the input documents
    self.docs_A           [Document1, ... ]
    self.docs_B           [Document1, ... ]
    self.annotators       set(['A', 'B', 'C', 'D'])
    self.training         {'A': <summary A>, ... }
    """

    def __init__(self, id, title, docs_A, docs_B):
        self.id = id
        self.title = title
        self.docs_A_paths = docs_A[:]
        self.docs_B_paths = docs_B[:]

        # for checking state
        self.loaded_docs = False
        self.parsed = False
        self.loaded_ir_docs = False

        # variables that might get set later
        self.docs_A = None
        self.docs_B = None
        self.training = {}
        self.annotators = set()

    def load_documents(self):
        self.docs_A = []
        self.docs_A = []
        for path in self.docs_A_paths:
            print(path)
            doc = text.Document(path)
            doc.get_sentences()
            self.docs_A.append(doc)

        self.docs_B = []
        for path in self.docs_B_paths:
            print(path)
            doc = text.Document(path)
            doc.get_sentences()
            self.docs_B.append(doc)

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
        for doc in self.docs_A:
            # count = 0
            for sent in doc.sentences:
                # cleaning
                if sent.original[0:2].islower():
                    print('bad parse:', sent.original)
                    continue
                sent_fh.write('%s\n' % sent.original)  # % cleaned)
                tok_fh.write('%s\n' % ' '.join(sent.tokens))
                lemm_fh.write('%s\n' % ' '.join(sent.lemm))
                # lemm_pos_fh.write('%s\n' % ' '.join(sent.lemm_pos))
                doc_fh.write('%s\n' % (doc.id))
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
        for doc in self.docs_B:
            # count = 0
            for sent in doc.sentences:
                # cleaning
                if sent.original[0:2].islower():
                    print('bad parse:', sent.original.encode(encoding))
                    continue
                sent_fh.write('%s\n' % sent.original)  # % cleaned)
                tok_fh.write('%s\n' % ' '.join(sent.tokens))
                lemm_fh.write('%s\n' % ' '.join(sent.lemm))
                # lemm_pos_fh.write('%s\n' % ' '.join(sent.lemm_pos))
                doc_fh.write('%s\n' % (doc.id))
                par_fh.write('%d\n' % int(sent.paragraph_starter))
        sent_fh.close()
        tok_fh.close()
        lemm_fh.close()
        # lemm_pos_fh.close()
        doc_fh.close()
        par_fh.close()

    def __str__(self):
        s = []
        s.append('%s SUMMARYPROBLEM' % '#START')
        s.append('ID %s' % self.id)
        s.append('TITLE %s' % self.title)
        s.append('NARR %s' % self.narr)
        s.append('DOCS_A %d\n%s' % (len(self.docs_A), '\n'.join(
            ['%s' % n for n in self.docs_A])))
        s.append('DOCS_B %d\n%s' % (len(self.docs_B), '\n'.join(
            ['%s' % n for n in self.docs_B])))
        for annotator in self.annotators:
            s.append('TRAIN %s\n%s' % (annotator, '\n'.join(
                ['%s' % n for n in self.training[annotator]])))
        return '\n'.join(s)


def setup_task(task):
    """
    task.topic_file: xml file for TAC
    task.doc_path: path containing source documents
    task.manual_path: path for manual (human) summaries
    """

    # get all document data
    all_docs = {}
    # print(task.doc_path)
    logger.debug(task.doc_path)
    files = util.get_files(task.doc_path,
                                    r'[^_]+_?[^_]*_?\d+[\.\-]\d+')
    logger.debug(files)
    # print(files)
    sys.stderr.write('Loading [%d] files\n' % len(files))
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
        # import pdb
        # pdb.set_trace()
        # print(all_docs)
        # print(doc_sets)
        for docset_index in range(0, len(docsets) - 1, 2):

            # map docids to documents
            docs_A = [all_docs[doc] for doc in docsets[docset_index]]
            docs_B = [all_docs[doc] for doc in docsets[docset_index+1]]

            # create a SummaryProblem
            problem = SummaryProblem(docset_ids[docset_index][:5],
                                     title, docs_A, docs_B)

            problems.append(problem)

    sys.stderr.write('Setting up [%d] problems\n' % len(problems))
    task.problems = problems


def setup_sentences(task, parser=None, reload=False, options=None):
    # load problems quickly from pickle file
    if (not reload) and os.path.isfile(task.data_pickle):
        sys.stderr.write('Loading [%s] problem data from [%s]\n'
                         % (task.name, task.data_pickle))
        task.problems = util.load_pickle(task.data_pickle)
        return

    # parse sentences
    for problem in task.problems:
        sys.stderr.write('%s\n' % problem.id)
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
    sys.stderr.write('Saving [%s] problem data in [%s]\n'
                     % (task.name, task.data_pickle))
    util.save_pickle(task.problems, task.data_pickle)
