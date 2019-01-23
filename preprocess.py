# -*- coding: utf-8 -*-

"""
    launch preprocess for documents define in topics_{task.name}

__author__ : Valentin Nyzam
"""

import time
import os
import preprocess.framework as prepframe
from globals import LOG_LEVEL
from globals import THREAD
from globals import ROOT
from globals import DATA_ROOT

class Task:
    """
    Class for holding paths to the important Task elements
    self.topic_file     xml/sgml file specifying problems (TAC/DUC)
    self.doc_path      path containing all source documents
    self.manual_path    path containing manual (human; gold) summaries
    """
    def __init__(self, task_name, topic_file, doc_path, manual_path=None,
                 length_limit=250):
        self.name = task_name
        self.topic_file = topic_file
        self.doc_path = doc_path
        self.manual_path = manual_path
        self.length_limit = length_limit
        self.data_pickle = '%s/%s_data.pickle' % (DATA_ROOT, self.name)
        self.problems = None


def parse_options():
    """
    set up command line parser options
    """
    from optparse import OptionParser
    usage = 'usage: %prog [options]'
    parser = OptionParser(usage=usage)

    parser.add_option('-t', '--task', dest='task', type='str',
                      help='task name for saving')
    parser.add_option('-i', '--docpath', dest='docpath', type='str',
                      help='source document path')
    parser.add_option('-l', '--length', dest='length', type='int',
                      help='maximum number of words in summaries')
    parser.add_option('-u', '--units', dest='units', default='n2', type='str',
                      help='units: n1 (unigrams), n2 (bigrams), \
                      su4 (skip bigrams + unigrams)')
    parser.add_option('--compress', dest='compress', default=False,
                      action='store_true', help='use sentence compression \
                      when generating summaries')
    parser.add_option('--reload', dest='reload', default=False,
                      action='store_true', help='reload document data from \
                      scratch')
    parser.add_option('-o', '--output', dest='output', default='%s/%s/out'
                      % (ROOT, 'output'), type='str',
                      help='output directory for summaries')
    (options, args) = parser.parse_args()

    if not options.docpath:
        parser.error('must specify a document path')

    topics_file = '%s/topics_%s' % (DATA_ROOT, options.task)

    task = Task(options.task, topics_file, options.docpath)

    # check valid units
    if options.units not in set(['n1', 'n2', 'n3', 'n4', 'su4']):
        parser.error('unrecognized unit selection [%s], use --help to get a \
                     list of valid units' % options.units)

    # create data root directory
    os.popen('mkdir -p %s' % DATA_ROOT)
    task.data_pickle = '%s/%s_data.pickle' % (DATA_ROOT, task.name)

    return options, task


def __init__():
    import logging
    from logging import StreamHandler
    from logging.handlers import TimedRotatingFileHandler

    # Create the Logger
    logger = logging.getLogger()
    logger.setLevel(LOG_LEVEL)

    if len(logger.handlers) > 0:
        logger.handlers.clear()

    # Create the Handler for logging data to a file
    logger_handler = TimedRotatingFileHandler('logs/preprocess.log',
                                              when='D', interval=1,
                                              backupCount=7)
    logger_handler.setLevel(logging.DEBUG)

    # Create the Handler for logging data to console.
    console_handler = StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Create a Formatter for formatting the log messages
    logger_formatter = logging.Formatter('%(name)s - %(threadName)s - %(levelname)s - %(message)s')

    # Add the Formatter to the Handler
    logger_handler.setFormatter(logger_formatter)
    console_handler.setFormatter(logger_formatter)

    # Add the Handler to the Logger
    logger.addHandler(logger_handler)
    logger.addHandler(console_handler)

    return logger

if __name__ == '__main__':
    logger = __init__()
    logger.info('Nb thread : ' + str(THREAD))
    options, task = parse_options()

    setup_start_time = time.time()

    # Load documents text and instance it as SummaryProblem
    prepframe.setup_task(task)

    # only run the parser if compression is required (this is not known by the
    # pickle stuff)
    parser = None
    if options.compress:
        logger.info('Using compression...')
    prepframe.setup_sentences(task, parser, reload=options.reload,
                              options=options)

    # get an empty directory
    path = os.path.join(ROOT, options.output)
    if os.path.isdir(path):
        os.system('rm -rf %s' % path)
    os.mkdir(path)
    # dump
    for problem in task.problems:
        problem.dump_data(path)
