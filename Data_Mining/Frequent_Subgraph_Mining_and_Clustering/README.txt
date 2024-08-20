Team Name: Data Crashers

Team Members:
1. Ajay Tomar (2023AIB2071): 33.33% contribution
2. Anshul Patel (2023AIB2072): 33.33% contribution
3. Pulkit Singal (2023AIB2064): 33.33% contribution



For Q1:


Explaination of all the bundled files:

1. format_f.cpp: C++ code for changing the format of the input dataset to the input format used by the executable file of fsg algorithm.
2. format_g.cpp: C++ code for changing the format of the input dataset to the input format used by the executable files of gSpan and gaston algorithm.
3. fsg: Executable file for fsg algorithm.
4. gaston: Executable file for gaston algorithm.
5. gSpan-64: Executable file for gSpan algorithm.
6. algo_time.py: Python script for writing the time corresponding to different algos from time.txt to final_time.txt
7. final_plot.py: Python script for generating the running time plots for different algos.
8. compile_Q1.sh: Script file for compiling the format_g.cpp and format_f.cpp files and for changing the permissions of the executable files gSpan-64, fsg and gaston.
9. run_Q1.sh: Script file for running the executable files corresponding to different algos, algo_time.py and final_plot.py


Instructions for executing the code:

1. Load the module compiler/gcc/9.1.0, if not already loaded by AIB232064_install.sh script file.
2. Install the python library matplotlib.
3. Run $ sh compile_Q1.sh
4. Run $ sh run_Q1.sh <input dataset>
5. The required plot q1_AIB232064.png will be generated.



For Q2:


Explaination of all the bundled files:

1. generateDataset_d_dim_hpc_compiled: Used to generate dataset by following the steps in Assignment2 instruction document.
2. Q2.py: Python code for reading the dataset file, implementation of k-means clustering algorithm and generating the elbow plot.
3. elbow_plot.sh: Script file for running Q2.py


Instructions for executing the code:

1. Generate dataset using generateDataset_d_dim_hpc_compiled by following the steps in Assignment 2 instruction document.
2. Install the python libraries pandas, matplotlib, scikit-learn (sklearn).
3. Run $ sh elbow_plot.sh <dataset> <dimension> q3_<dimension>_<RollNo>.png
4. The required output .png elbow plot file will be generated.