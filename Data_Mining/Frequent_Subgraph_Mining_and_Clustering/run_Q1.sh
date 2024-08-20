#!/bin/bash

./format_g $1
./format_f $1

for minSup_gSpan in 0.05 0.1 0.25 0.5 0.95
do
	/usr/bin/time -p -o time.txt ./gSpan-64 -f "format_g.txt_graph" -s $minSup_gSpan -o
	python3 algo_time.py
done

for minSup_fsg in 5 10 25 50 95
do
	/usr/bin/time -p -o time.txt ./fsg -s $minSup_fsg "format_f.txt_graph"
	python3 algo_time.py
done

for minSup_gaston in 3205.5 6411 16027 32055 60904.5
do
	/usr/bin/time -p -o time.txt ./gaston $minSup_gaston "format_g.txt_graph" "gaston_out.txt_graph"
	python3 algo_time.py
done

python3 final_plot.py