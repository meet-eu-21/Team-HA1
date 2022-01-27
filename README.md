# MinCutTAD: Interpretable graph neural network - driven TAD prediction from Hi-C chromatin interactions and chromatin states

## Abstract

<ul>
<li>GNN algorithm driven by spectral clustering to detect TADs. Constructed with GraphConv, a message passing layer, and if the algorithm is unsupervised with a MinCut pooling layer.</li>
<ul>
<li>Message passing refers to the smoothening of the information among the directly surrounding node features.</li>
<li>Pooling refers to the aggregation of strongly similar nodes, thereby reducing the graph domain and forming sub clusters.</li>
</ul>
<li>Utilizes Hi-C matrices data & genomic annotations (CTCF, RAD21, SMC3, # of housekeeping genes) for the provided genomic loci of chromosomes</li>
</ul>

Two approaches: 
<ul>
<li>Supervised uses Arrowhead solutions as labels for the genomic bins and optimizes towards classifying the graph nodes accordingly to those.</li>
<li>Unsupervised: no labels are provided to the model, and it determines whether regions belong to a TAD or not and aggregate them. Therefore, its main goal is to cluster single TAD regions together.</li>
</ul>



## Repository Structure 

The folder structure of the repsoitory is shown below. The folders ```./TopResults```, ```./cmap_files```, ```./node_annotations``` and  ```./ressources``` contain files necessary for running the scripts in the folder ```./tad_detection```.

<pre>
├── TopResults
│   └── 100kb100kb
│       └── GM12878
├── cmap_files
│   ├── 25kb
│   │   ├── GM12878
│   │   │   └── intra
│   │   └── IMR-90
│   │       └── intra
│   └── 100kb
│       ├── GM12878
│       │   ├── inter
│       │   └── intra
│       └── IMR-90
│           └── intra
├── node_annotations
├── ressources
├── tad_detection
│   ├── evaluation
│   ├── model
│   ├── preprocessing
│   └── utils_general.py
├── LICENSE
├── README.md
└── environment.yml
</pre>

The scripts developed as part of this project can be found in the folder ```./tad_detection``` and the corresponding subfolders.

An exact description of the preprocessing scripts can be found in the folder [`./tad_detection/preprocessing`](tad_detection/preprocessing) and the associated [README](tad_detection/preprocessing/README.md).<br>
An exact description of the training scripts can be found in the folder [`./tad_detection/model`](tad_detection/model) and the associated [README](tad_detection/model/README.md).<br>
An exact description of the evaluation scripts can be found in the folder [`./tad_detection/evaluation`](tad_detection/evaluation) and the associated [README](tad_detection/evaluation/README.md).<br>
An exact description of the benchmarking tools scripts can be found in the folder [`./tad_detection/evaluation/tools_benchmarking`](tad_detection/evaluation/tools_benchmarking) and the associated [README](tad_detection/evaluation/tools_benchmarking/README.md).<br>


## Running the tools in this repository

The tools must be run with `./MeetEU` as the working directory. An [`environment.yml`](environment.yml) file with a list of all the necessary packages for our model and scripts is available in the repository. Please note that some of the packages may only be available for UNIX-based operating systems. The usage of a HPC with access to a GPU is highly recommended for the training of the model.

## Sample data

Sample data to run this algorithm can be found [here](https://1drv.ms/u/s!AvJpVBIXzqzAiNQ8xRJap-PdlQOtgQ?e=wEwyH5).
