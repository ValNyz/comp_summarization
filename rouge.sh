#!/bin/bash
DOCS=/home/valnyz/PhD/data/comparative/eval_doc
REF=/home/valnyz/PhD/data/comparative/model_sum
ROUGE_HOME=/home/valnyz/PhD/ROUGE-1.5.5/RELEASE-1.5.5
#TASK=$1
#OUTPUT="output/$TASK/"

python3 ./rouge_eval.py -r $ROUGE_HOME -m $REF -p $1
