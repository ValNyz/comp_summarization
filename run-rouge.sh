#!/bin/bash
DOCS=/home/nyzam/data/comparative/eval_doc
REF=/home/nyzam/data/comparative/model_sum
ROUGE_HOME=/home/nyzam/ROUGE-1.5.5/RELEASE-1.5.5
TASK=$1
OUTPUT="output/$TASK/"
#export PYTHONPATH=splitta:$PYTHONPATH
#echo $PYTHONPATH
#nltk/nltk-0.9.2:
#export PATH=solver/glpk-4.43/examples/:$PATH
mkdir -p $OUTPUT

python3 rouge_eval.py -r $ROUGE_HOME -m $REF -p $OUTPUT/summary
