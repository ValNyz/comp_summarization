#!/bin/bash
DOCS=/home/arch/valnyz/PhD/data/comparative
REF=/home/arch/valnyz/PhD/data/comparative
TASK=$1
OUTPUT="output/$TASK/"
#export PYTHONPATH=splitta:$PYTHONPATH
#echo $PYTHONPATH
#nltk/nltk-0.9.2:
#export PATH=solver/glpk-4.43/examples/:$PATH
mkdir -p $OUTPUT

python3 preprocess.py -i $DOCS -o $OUTPUT --task $TASK --reload

#for i in $OUTPUT/*.sent ; do preprocess/penn_treebank_tokenizer.sed $i > $i.tok;done

#export HOSTNAME=localhost
#python2 summarizer/inference.py -i $OUTPUT -o $OUTPUT -r tac -t $TASK --decoder nbest --nbest 1 #--manpath $REF
#rm -f tmp_decoder.*
