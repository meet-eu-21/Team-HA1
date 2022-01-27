# Model

This folder contains the functions and PyTorch layers and models for our MinCutTAD model. It is the core of our project.
The folder structure is shown below. The main scripts, which can be run independently, are marked. Below the purpose of each main script is discussed and it is described how to call each of these scripts.

<pre>
    └── model
        ├── gatconv.py
        ├── gnn_explainer.py
        ├── <b><em>metrics.py</em></b>
        ├── mincuttad.py
        ├── parameters.json
        ├── <b><em>train.py</em></b>
        └── utils_model.py
</pre>

```gatconv.py``` provides a slightly modified version of the GATConvv2 implementation in  PyTorch Geometric.<br>
```gnn_explainer.py``` provides a slightly modified version of the GNN Explainer implementation in  PyTorch Geometric.<br>
The functions in ```metrics.py``` are used to calculate metrics such as the ROC score and the silhouette score, log them and create plots.<br>
```mincuttad.py``` provides a slightly modified version of the GNN Explainer implementation in  PyTorch Geometric.<br>
The functions in ```utils_model.py``` are used as helper functions to load and split data, to load and save models, provide logging and generate TAD regions.<br>

## Scripts

### train.py

```
usage: train.py [-h] [--path_parameters_json PATH_PARAMETERS_JSON]

Train and evaluate a model on dataset created in preprocessing pipeline.

optional arguments:
  -h, --help            show this help message and exit
  --path_parameters_json PATH_PARAMETERS_JSON
                        path to JSON with parameters.

```

The results of this script are provided in the dataset results folder.

### metrics.py

```
usage: metrics.py [-h] [--path_parameters_json PATH_PARAMETERS_JSON]

Create ROC curve plots and silhouette score plots for multiple experiments
together.

optional arguments:
  -h, --help            show this help message and exit
  --path_parameters_json PATH_PARAMETERS_JSON
                        path to JSON with parameters.
```

The results of this script are provided in the dataset results folder. Running this script assumes that at least one experiment has been run using train.py.

## Parameters

The ```parameters.json``` file contains the used parameters for the different functions in this folder.  In the parameters.json file several variables can be set, which will be described below:

```
parameters.json

variables:
  activation_function: activation function DL
                                        "Relu"
  attention_heads_num: number of heads GATConv layer
                                        2
  dataset_path: directory of dataset
                                        "./data/"
  dataset_name: name of dataset
                                        "dataset_name"
  classes: binary classes
                                        ["TAD", "No_TAD"]
  comparison_metrics_datasets: names of dataset , which are compared in metrics.py
                                        ["dataset_1", "dataset_2"]
  comparison_metrics_datasets_labels: labels of dataset , which are compared in metrics.py
                                        ["dataset_1_label", "dataset_2_label"]
  comparison_metrics_name: name of metrics comparison experiment              
                                        "dataset_1_vs_dataset_2"
  epoch_num: number of epochs for training                
                                        2, 10, 97
  encoding_edge: message passing via GCNConv or GraphConv for graph classification training   
                                        true, false
  hidden_conv_size: hidden size of convolutional layer for graph classification training
                                        32
  learning_rate: learning rate for training 
                                        0.00005, 0.001, 0.03
  learning_rate_decay_patience: learning rate decay patience for training 
                                        10
  n_channels: number of channels message passing
                                        16
  num_layers: number of layers for graph classification training   
                                        4
  num_epochs_gnn_explanations: number of epochs for training model for GNN node explanations
                                        20
  message_passing_layer: type of message passing layer for training
                                        "GraphConv", "GATConv"
  genomic_annotations: genomic annotations used in dataset creation
                                        ["CTCF", "RAD21", "SMC3", "housekeeping_genes"]
  generate_gnn_explanations: boolean whether GNN node expalantions should be created
                                        false, true
  generate_graph_matrix_representation: boolean whether graph matrix representation should be calculated before training
                                        false, true
  graph_matrix_representation: type of graph matrix representation
                                        "laplacian"
  optimizer: optimizer fir training
                                        "adam"
  output_directory: directory where models, metrics, plots, predicted TADs and node explanations are saved
                                        "./results/"
  pooling_type: pooling type for graph classification training   
                                        "linear", "random"
  proportion_train_set: proportion of training set for training
                                        0.6
  proportion_test_set: proportion of test set for training
                                        0.2
  proportion_val_set: proportion of validation set for training
                                        0.2
  scaling_factor: resolution of Hi-C adjacency matrix
                                        100000
  task_type: task type for training
                                        "supervised", "unsupervised"
  weight_decay: weight decay for training
                                        0.05
```
