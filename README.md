# MinCutTAD: Graph neural network - driven hierarchical TAD boundary prediction from HiC chromatin interaction and chromatin state data

The preprocessing of 

An exact description of the preprocessing scripts can be found in the folder ```./preprocessing``` and the associated [README](code/preprocessing/README.md).

An exact description of the training scripts can be found in the folder ```./model``` and the associated [README](code/model/README.md).

An exact description of the evaluation scripts can be found in the folder ```./evaluation``` and the associated [README](code/model/README.md).




This repository contains the node annotations for the datasets provided by Rao et al. 

The list of housekeeping genes is provided in the file ```./ressources/Housekeeping_GenesHuman.csv```.
The housekeeping genes were published in the HRT Atlas (Hounkpe et al., 2021). The named file is provided in the GitHub reposiory of the [HRT Atlas](https://github.com/Bidossessih/HRT_Atlas/tree/master/www).

We downloaded chromatin state stets based on the assays for CTCF, RAD21 and SMC3 from the ENCODE portal (Sloan et al., 2016; Davis et al., 2018) (https://www.encodeproject.org/) with the following identifiers: 
<ol>
<li><b>Cell line GM12878:</b> ENCFF074FXJ (CTCF), ENCFF110OBQ (RAD21) and ENCFF049WIK (SMC3)</li>
<li><b>Cell line IMR-90:</b> ENCFF276MRX (CTCF), ENCFF374EXW (RAD21) and ENCFF476RFS (SMC3)</li>
</ol>

Several tools have been 
<ol>
<li>Basic spectral clustering based on scikit-learn (Pedregosa et al., 2011)</li>
<li>SpectralTAD (Cresswell et al., 2020)</li>
<li>(Ashoor et al., 2020)</li>
<li>TopDom (Shin et al., 2016)</li>
<ol>



An ````environment.yml```` file with a list of all the necessary python packages for our scripts has been provided. For the execution of the benchmarking tools please refer to the respective documentation.

## Ressources

https://academic.oup.com/nar/article/49/D1/D947/5871367
https://pubmed.ncbi.nlm.nih.gov/26527727/
https://pubmed.ncbi.nlm.nih.gov/29126249/