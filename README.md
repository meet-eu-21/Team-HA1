# MinCutTAD: Graph neural network - driven hierarchical TAD boundary prediction from HiC chromatin interaction and chromatin state data

## Description
An exact description of the preprocessing scripts can be found in the folder [`./tad_detection/preprocessing`](tad_detection/preprocessing) and the associated [README](tad_detection/preprocessing/README.md).

An exact description of the training scripts can be found in the folder [`./tad_detection/model`](tad_detection/model) and the associated [README](tad_detection/model/README.md).

An exact description of the evaluation scripts can be found in the folder [`./tad_detection/evaluation`](tad_detection/evaluation) and the associated [README](tad_detection/evaluation/README.md).

The dictionaries containing the genomic annotations (CTCF, RAD21, SMC3 and housekeeping genes) for the datasets provided by Rao et al. can be found in the folder [`./node_annotations`](node_annotations).

The list of housekeeping genes is provided in the file [`./ressources/Housekeeping_GenesHuman.csv`](ressources/Housekeeping_GenesHuman.csv).
The housekeeping genes were published in the HRT Atlas (Hounkpe et al., 2021). The named file is provided in the GitHub reposiory of the [HRT Atlas](https://github.com/Bidossessih/HRT_Atlas/tree/master/www).

We downloaded chromatin state sets based on the assays for CTCF, RAD21 and SMC3 from the ENCODE portal (Sloan et al., 2016; Davis et al., 2018) (https://www.encodeproject.org/) with the following identifiers: 
1. **Cell line GM12878**: [ENCFF074FXJ (CTCF)](https://www.encodeproject.org/annotations/ENCFF074FXJ/), [ENCFF110OBQ (RAD21)](https://www.encodeproject.org/annotations/ENCFF110OBQ/) and [ENCFF049WIK (SMC3)](https://www.encodeproject.org/annotations/ENCFF049WIK/)
1. **Cell line IMR-90**: [ENCFF276MRX (CTCF)](https://www.encodeproject.org/annotations/ENCFF276MRX/), [ENCFF374EXW (RAD21)](https://www.encodeproject.org/annotations/ENCFF374EXW/) and [ENCFF476RFS (SMC3)](https://www.encodeproject.org/annotations/ENCFF476RFS/)


Several tools have already been developed for this purpose, which we used as inspiration and partly as evaluation tools:
1. Arrowhead (Rao et al., 2014)
1. TopDom (Shin et al., 2016)
1. Basic spectral clustering based on scikit-learn (Pedregosa et al., 2011)
1. SpectralTAD (Cresswell et al., 2020)
1. (Ashoor et al., 2020)

For the execution of the benchmarking tools please refer to the respective documentation.

An [`environment.yml`](environment.yml) file with a list of all the necessary python packages for our scripts has been provided. 

## Sample data
The sample data to run this algorithm can be found [here](https://1drv.ms/u/s!AvJpVBIXzqzAiNQ8xRJap-PdlQOtgQ?e=wEwyH5).

## Ressources

https://academic.oup.com/nar/article/49/D1/D947/5871367 <br>
https://pubmed.ncbi.nlm.nih.gov/26527727/ <br>
https://pubmed.ncbi.nlm.nih.gov/29126249/
