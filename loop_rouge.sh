#!/bin/bash

touch result
for i in `LC_ALL=C seq 0.5 0.05 1`; do
	echo $i >> result
	./rouge.sh output/comp/summary/knapsack2_WE_SEN_WMD/$i
	/home/valnyz/PhD/ROUGE-1.5.5/RELEASE-1.5.5/ROUGE-1.5.5.pl -e /home/valnyz/PhD/ROUGE-1.5.5/RELEASE-1.5.5/data -n 3 -x -m -c 95 -r1000 -f A -p 0.5 -t 0 -a output/comp/summary/knapsack2_WE_SEN_WMD/rouge_settings.xml >> result
	sleep 1
done
