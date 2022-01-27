# Preprocessing

This folder contains the different notebooks and functions used to generate the cmap files, perform the preprocessing, create visualizations and prepare the datasets for the ```model``` and the ```evaluation```.
The folder structure is shown below. The main scripts, which can be run independently, are marked. Below the purpose of each main script is discussed and it is described how to call each of these scripts.

<pre>
└── tad_detection
    └── preprocessing
        ├── <b><em>arrowhead_solution.py</em></b>
        ├── <b><em>cmap_generator.ipynb</em></b>
        ├── <b><em>chr_len_dict.py</em></b>
        ├── graph_generation.py
        ├── <b><em>intra_inter_comparison.ipynb</em></b>
        ├── <b><em>node_annotations.py</em></b>
        ├── parameters.json
        ├── <b><em>preprocessing.py</em></b>
        ├── utils_preprocessing.py
        └── visualization.py
</pre>

The notebook ```ccmap_generator.ipynb``` can be used to create the cmap files containing the Hi-C adjacency matrices used as input for all preprocessing scripts. These adjacency matrices are created using the data provided for this project. The cmap files are automatically saved in the folder ```./cmap_files```. All cmap files are available for the cell lines GM12878 and IMR-90 for a resolution of 25kb. For the resolution of 100kb the cmap files for the first three chromosomes are provided due to memory constraints on GitHub. The remaining cmap files can be created via the notebook or available upon request.<br>
The functions in ```arrowhead_solution.py``` load the TAD classification of Arrowhead and create a one-hot encoded list (1 for TAD, 0 for non-TAD) for all nodes in an adjacency matrix.<br>
The functions in ```graph_generation.py``` load the adjacency matrices from raw data, filter the edges and vertices of the resultig graph by a certain threshold and generate statistics for the created adjacency matrices.<br>
The functions in ```utils_preprocessing.py``` are used as helper functions to generate a list of chromosomes, the edge index and edge attributes of adjacency matrices.<br>
The ```node_annotations.py```file contains all main functions for the generation of the node features.
1. The genomic annotations we used are downloaded from ENCODE for the 2 cell lines GM12878 ([CTCF](https://www.encodeproject.org/annotations/ENCFF074FXJ/), [RAD21](https://www.encodeproject.org/annotations/ENCFF110OBQ/) and [SMC3](https://www.encodeproject.org/annotations/ENCFF049WIK/)) and IMR-90 ([CTCF](https://www.encodeproject.org/annotations/ENCFF276MRX/), [RAD21](https://www.encodeproject.org/annotations/ENCFF374EXW/) and [SMC3](https://www.encodeproject.org/annotations/ENCFF476RFS/)).<br> In ```generate_dict_genomic_annotations``` the signal strength of annotations in a specified range are added up to generate a dictionary with the number of annotations for each bin. This dictionary can then be loaded with ```load_dict_genomic_annotations```.
2. The list of housekeeping genes is provided in the file ```./ressources/Housekeeping_GenesHuman.csv```which can be loaded using the function ```load_housekeeping_genes```.The housekeeping genes were published in the HRT Atlas (Hounkpe et al., 2021). The named file is provided in the GitHub reposiory of the [HRT Atlas](https://github.com/Bidossessih/HRT_Atlas/tree/master/www).<br> In ```generate_dict_housekeeping_genes``` a one-hot enoded doctionary is generated, which saves if a housekeeping gene is present in a certain genomic bin or not. This dictionary can then be loaded with `load_dict_housekeeping_genes`.
3. Lastly, in `combine_genomic_annotations_and_housekeeping_genes` a list of annotation matrices is constructed, where the dictionary of the genomic annotations and housekeeping genes are combined for every indicated chromosome for both cell lines.

## Scripts

### arrowhead_solution.py

```
usage: arrowhead_solution.py [-h] --path_parameters_json PATH_PARAMETERS_JSON

Create numpy array and dicts of true labels called by Arrowhead for genomic
bins for chosen chromosomes and cell lines.

optional arguments:
  -h, --help            show this help message and exit
  --path_parameters_json PATH_PARAMETERS_JSON
                        path to JSON with parameters.
```

The results of this script are provided in the ```./evaluation/results``` folder.

### node_annotations.py

```
usage: node_annotations.py [-h] --path_parameters_json PATH_PARAMETERS_JSON

Create dictionaries of genomic annotations and occurrence of housekeeping
genes for genomic bins used in preprocessing pipeline used for dataset
creation.

optional arguments:
  -h, --help            show this help message and exit
  --path_parameters_json PATH_PARAMETERS_JSON
                        path to JSON with parameters.
```

The results of this script are provided in the ```./ressources``` folder.

### chr_len_dict.py

```
usage: chr_len_dict.py [-h] --path_parameters_json PATH_PARAMETERS_JSON

Create dictionary of chromosome lengths for usage in evaluation pipeline.

optional arguments:
  -h, --help            show this help message and exit
  --path_parameters_json PATH_PARAMETERS_JSON
                        path to JSON with parameters.

```

The results of this script are already provided in the ```./ressources``` folder.

### preprocessing.py

```
usage: preprocessing.py [-h] --path_parameters_json PATH_PARAMETERS_JSON

Create dataset consisting out of edge_index, edge_attr, source_information and
labels from cmap files, genomic annotations and housekeeping dicts and
arrowhead solution or the labels used by the french team.

optional arguments:
  -h, --help            show this help message and exit
  --path_parameters_json PATH_PARAMETERS_JSON
                        path to JSON with parameters.

```

The script assumes that node_annotations.py has been run before or that the results already are saved in the ```./ressources``` folder.


## Parameters

The ```parameters.json``` file contains the used parameters for the different functions in this folder.  In the parameters.json file several variables can be set, which will be described below:

```
parameters.json

variables:
  arrowhead_solution_GM12878: path to labels of arrowhead solution for GM12878 cell line 
                                                    "./data/www.lcqb.upmc.fr/meetu/dataforstudent/TAD/GSE63525_GM12878_primary+replicate_Arrowhead_domainlist.txt"
  arrowhead_solution_IMR-90: path to labels of arrowhead solution for GM12878 cell line 
                                                    "./data/www.lcqb.upmc.fr/meetu/dataforstudent/TAD/GSE63525_IMR90_Arrowhead_domainlist.txt"
  binary_dict: usage of a binary dict (0.25 quantile, 0.5 quantile, 0.75 wquantile) for the creation of node annotations in preprocessing.py
                                                    "False", "True"
  cell_lines: cell lines, for which dataset in preprocessing.py or node annotations node_annotations.py or chromosome length dict in chr_len_dict.py should be created
                                                    ["GM12878", "IMR-90"]
  chromosomes: chromosomes, for which dataset in preprocessing.py or node annotations node_annotations.py or chromosome length dict in chr_len_dict.py should be created
                                                    "all", ["1", "2", ...]
  dataset_name: name of dataset created in preprocessing.py
                                                    "dataset_name",
  ensembl_gtf_file_path: path to gtf file for housekeeping_genes dict generation in node_annotations.py
                                                    "./data/ensembl/Homo_sapiens.GRCh37.55.gtf"
  ensembl_gtf_database_path: path to gtf database file for housekeeping_genes dict generation in node_annotations.py
                                                    "./data/ensembl/Homo_sapiens.GRCh37.55.db"
  genomic_annotations: node annotations, which are incorporated in dataset created in preprocessing.py
                                                    ["CTCF", "RAD21", "SMC3", "housekeeping_genes"]
  genomic_annotations_directory: directory of genomic annotation files (CTCF, RAD21, SMC3)
                                                    "./data/encode_data"
  genomic_annotations_CTCF_GM12878: path bigWig file CTCF genomic annotations for GM12878 cell line
                                                    "./data/encode_data/ENCFF074FXJ.bigWig"
  genomic_annotations_CTCF_IMR-90: path bigWig file CTCF genomic annotations for IMR-90 cell line
                                                    "./data/encode_data/ENCFF276MRX.bigWig"
  genomic_annotations_RAD21_GM12878: path bigWig file RAD21 genomic annotations for GM12878 cell line
                                                    "./data/encode_data/ENCFF110OBQ.bigWig"
  genomic_annotations_RAD21_IMR-90: path bigWig file RAD21 genomic annotations for IMR-90 cell line
                                                    "./data/encode_data/ENCFF374EXW.bigWig"
  genomic_annotations_SMC3_GM12878: path bigWig file SMC3 genomic annotations for GM12878 cell line
                                                    "./data/encode_data/ENCFF049WIK.bigWig"
  genomic_annotations_SMC3_IMR-90: path bigWig file SMC3 genomic annotations for IMR-90 cell line
                                                    "./data/encode_data/ENCFF476RFS.bigWig"
  genomic_annotations_dicts_directory: directory of dicts of node annotation dicts with genomic annotations (CTCF, RAD21, SMC3) and the housekeeping genes dict
                                                    "./node_annotations/"
  hic_matrix_directory: directory of HiC matrices used in ccmap_generator.ipynb
                                                    "./www.lcqb.upmc.fr/meetu/dataforstudent/HiC/"
  label_types: types of true labels used for dataset generation
                                                    "arrowhead", "french_team_labels"
  path_labels_french_team: path to .csv file with TAD boundaries for true label generation for labels of french team
                                                    "./ressources/annotated_position.csv"
  node_feature_encoding: genomic annotations used in dataset creation
                                                    ["CTCF", "RAD21", "SMC3", "Number_HousekeepingGenes"]
  output_directory: output directory, where the dataset is saved
                                                    "./data/"
  path_housekeeping_genes: path of housekeeping genes ressource, from which the node annotations are generated
                                                    "./ressources/Housekeeping_GenesHuman.csv"
  quantile_genomic_annotations: quantiles of genomic annotations used to load bianrized versions of the genomic annotations if binary_dict is "True"
                                                    0.25, 0.5, 0.75
  resolution_hic_matrix: resolution of Hi-C adjacency matrix
                                                    25000, 100000
  resolution_hic_matrix_string: resolution of Hi-C adjacency matrix
                                                    "25kb", "100kb"
  scaling_factor: resolution of Hi-C adjacency matrix
                                                    25000, 100000
  generate_plots_statistics: bool-type whether statistics - Hi-C map visualizations, quantiles etc. should be generated and calculated
                                                    "False", "True"
  graph_vertex_filtering_min_val: minimum amount of vertices a genomic needs to have before it is filtered
                                                    "None", 10, 20, ...
  threshold_graph_vertex_filtering: threshold for edge so the count of edges by genomic bin can be evalued comparing with graph_vertex_filtering_min_val
                                                    "None", 5, 8, ...
  threshold_graph_edge_filtering: threshold for filtering edges from adjacency matrix                    
                                                    "None", 5, 6, ...
  translation_ensembl_geneid_to_external_geneid: mapping file to translate a gene ID to an external gene ID used in the generation of housekeeping genes dict
                                                    "./ressources/translation_ensembl_gene_to_external_gene_name.csv"
```




