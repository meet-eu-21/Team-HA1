import json
import logging
import os
import numpy as np

def load_parameters(path_parameters_json):
    with open(path_parameters_json) as parameters_json:
        parameters = json.load(parameters_json)

    return parameters

def set_up_logger(parameters):
    logger = logging.getLogger('preprocessing')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(parameters["output_path"], 'preprocessing', 'preprocessing.log'))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    logger.info("Training with the following parameters:")
    for parameter in parameters.keys():
        logger.info(parameter + ": " + str(parameters[parameter]))


def generate_chromosome_lists(parameters):
    if parameters["chromosomes"] == "all":
        chromsomes_int = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, "X"]
        chromosomes_str_long = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10",
                                "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19",
                                "chr20", "chr21", "chr22", "chrX"]
        chromosomes_str_short = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16",
                                 "17", "18", "19", "20", "21", "22", "X"]
    else:
        chromsomes_int = parameters["chromosomes"]
        chromosomes_str_long = []
        chromosomes_str_short = []
        for chromosome in parameters["chromosomes"]:
            chromosomes_str_long.append("chr" + str(chromosome))
            chromosomes_str_short.append(str(chromosome))

    return chromsomes_int, chromosomes_str_long, chromosomes_str_short

'''
def extract_bins(graph_matrices, chromosome_list_graph_matrices):

    :param graph_matrices: adjacency matrices representing Hi-C graphs for a specific chromosome and cell line
    :param chromosome_list_graph_matrices: list of chromosome names for each graph in graph_matrices
    :return bins_chromosomes: bins for each chromosome

    bins_chromosomes = []

    for chromsome, graph in zip(chromosome_list_graph_matrices, graph_matrices):
        graph.rows
'''

def save_adjacency_matrix_node_features_labels(parameters, graph, node_features, arrowhead_solution):

    np.save(node_features, os.path.join(parameters["output_directory"], parameters["dataset_name"] + "_X.npy"))
    np.save(graph, os.path.join(parameters["output_directory"], parameters["dataset_name"] + "_edge_index.npy"))
    np.save(arrowhead_solution, os.path.join(parameters["output_directory"], parameters["dataset_name"] + "_y.npy"))