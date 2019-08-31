#!/bin/bash

rm -rf tf_scripts/suggester/cases_csv && rm -f tf_scripts/suggester/process.csv && python3 tf_scripts/suggester/generate_data.py && python3 tf_scripts/suggester/elbow.py && echo "How many clusters should be created?" && read num && python3 tf_scripts/suggester/kmeans.py -n $num && python3 tf_scripts/suggester/footprint_cases.py && python3 tf_scripts/suggester/pos_histogram.py

for ((i=0;i<=(num-1);i++));
  do python3 tf_scripts/suggester/rnn_new.py -f $i -s $i; 
done
