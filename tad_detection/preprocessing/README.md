# Preprocessing
This folder contains the different functions used to perform the preprocessing.

The functions in ```arrowhead_solution.py``` load the TAD classification of Arrowhead and create a one-hot encoded list (1 for TAD, 2 for non-TAD) for all nodes in an adjacency matrix.

The functions in ```graph_generation.py``` load the adjacency matrices from raw data, filter the edges and vertices of the resultig graph by a certain threshold and generate statistics for the created adjacency matrices.

The ```node_annotations.py```file contains all main functions for the genertion of the node features.
1. The genomic annotations we used are downloaded from ENCODE for the 2 cell lines GM12878 ([CTCF](https://www.encodeproject.org/annotations/ENCFF074FXJ/), [RAD21](https://www.encodeproject.org/annotations/ENCFF110OBQ/) and [SMC3](https://www.encodeproject.org/annotations/ENCFF049WIK/)) and IMR-90 ([CTCF](https://www.encodeproject.org/annotations/ENCFF276MRX/), [RAD21](https://www.encodeproject.org/annotations/ENCFF374EXW/) and [SMC3](https://www.encodeproject.org/annotations/ENCFF476RFS/)).<br> In ```generate_dict_genomic_annotations``` the signal strength of annotations in a specified range are added up to generate a dictionary with the number of annotations for each bin. This dictionary can then be loaded with ```load_dict_genomic_annotations```.
2. The list of housekeeping genes is provided in the file ```./ressources/Housekeeping_GenesHuman.csv```which can be loaded using the function ```load_housekeeping_genes```.The housekeeping genes were published in the HRT Atlas (Hounkpe et al., 2021). The named file is provided in the GitHub reposiory of the [HRT Atlas](https://github.com/Bidossessih/HRT_Atlas/tree/master/www).<br> In ```generate_dict_housekeeping_genes``` a one-hot enoded doctionary is generated, which saves if a housekeeping gene is present in a certain genomic bin or not. This dictionary can then be loaded with `load_dict_housekeeping_genes`.
3. Lastly, in `combine_genomic_annotations_and_housekeeping_genes` a list of annotation matrices is constructed, where the dictionary of the genomic annotations and housekeeping genes are combined for every indicated chromosome for both cell lines.

The functions in ```utils_preprocessing.py``` are used as helper functions to generate a list of chromosomes, the edge index and edge attributes of adjacency matrices.

The ```parameters.json``` file contains the used parameters for the different functions in this folder. 



