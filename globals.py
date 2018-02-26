# -*- coding: utf-8 -*-

"""
global paths for shared use, change the ROOT to your installation directory
__author__ : Valentin Nyzam
"""

import os

THREAD = 8
#ROOT = os.path.realpath(os.path.dirname(sys.argv[0])) + '/../'
THREAD = 8
ROOT = '/home/nyzam/comp_summarization'

DATA_ROOT = os.path.join(ROOT, 'data')
TOOLS_ROOT = os.path.join(ROOT, 'tools')

STOPWORDS = os.path.join(DATA_ROOT, 'stopwords.english')
BERKELEY_PARSER_CMD = '%s/parser_bin/distribute.sh %s/parser_bin/berkeleyParser+Postagger.sh' %(TOOLS_ROOT, TOOLS_ROOT)
