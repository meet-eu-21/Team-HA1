# Tools benchmarking

This folder contains the different functions to run different benchmarking tools for the comparison with the results of our model.
The folder structure is shown below. The main scripts, which can be run independently, are marked. Below the purpose of each main script is discussed and it is described how to call each of these scripts.

<pre>
└── tad_detection
    └── evaluation
        └── tools_benchmarking
            ├── TopDomResults
            ├── <b><em>TopDom_Run.R</em></b>
            ├── <b><em>TopDom_preprocessing.py</em></b>
            └── <b><em>simple_spectral_clustering.py</em></b>
</pre>

## TopDom_Run.R
Given the extended adjacency matrices, the TopDom package detects TADs and results are saved in .bed, .binsignal, and .domain files. 

## TopDom_preprocessing.py
Turns given Hi-C data into a normalized adjacency matrix with extra three columns: chrnum, binstart, binend. Saves adjacency
matrix as a csv file, to be used in the R script TopDom_Run.R

