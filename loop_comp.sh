#!/bin/bash

for i in `LC_ALL=C seq 0.2 0.05 0.5`; do
	echo $i
	python comp_summarization.py -t $i -k comp -m WE_SEN_WMD -c knapsack2
done
