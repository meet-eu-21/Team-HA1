import sys
sys.path.insert(1, './tad_detection/')
sys.path.insert(1, './tad_detection/preprocessing/')
sys.path.insert(1, './tad_detection/model/')
sys.path.insert(1, './tad_detection/evaluation/')

import logging
logger = logging.getLogger('preprocessing')

import json
import logging
import os
import numpy as np

def _old_load_parameters(path_parameters_json):
    '''
    Function loads the parameters from the provided parameters.json file in a dictionary.

    :param path_parameters_json: path of parameters.json file
    :return parameters: dictionary with parameters set in parameters.json file.
    '''
    with open(path_parameters_json) as parameters_json:
        parameters = json.load(parameters_json)

    return parameters

def _old_set_up_logger(parameters):
    '''
    Function sets a global logger for documentation of information and errors in the execution of the chosen script.

    :param parameters: dictionary with parameters set in parameters.json file.
    '''

    logger = logging.getLogger('preprocessing')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(parameters["output_directory"], 'preprocessing', 'preprocessing.log'))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    logger.info("Training with the following parameters:")
    for parameter in parameters.keys():
        logger.info(parameter + ": " + str(parameters[parameter]))


def generate_chromosome_lists(parameters):
    '''

    :param parameters: dictionary with parameters set in parameters.json file
    :return:
    '''
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

def generate_edge_index_edge_attr_from_adjacency_matrix(parameters, adjacency_matrices_list):

    edge_index = []
    edge_attr = []

    for cell_line, adjacency_matrices_list_cell_line in zip(parameters["cell_lines"], adjacency_matrices_list):
        edge_index_cell_line = []
        edge_attr_cell_line = []
        for adjacency_matrix in adjacency_matrices_list_cell_line:
            edge_index_cell_line_chromosome_1, edge_index_cell_line_chromosome_2 = np.where(adjacency_matrix > 1)
            edge_index_cell_line.append([edge_index_cell_line_chromosome_1, edge_index_cell_line_chromosome_2])
            edge_attr_cell_line_chromosome = []
            for edge_index_1, edge_index_2 in zip(edge_index_cell_line_chromosome_1, edge_index_cell_line_chromosome_2):
                edge_attr_cell_line_chromosome.append(adjacency_matrix[edge_index_1, edge_index_2])
            edge_attr_cell_line.append(np.array(edge_attr_cell_line_chromosome))
        edge_index.append(np.array(edge_index_cell_line))
        edge_attr.append(np.array(edge_attr_cell_line))

    return edge_index, edge_attr

def save_adjacency_matrix_node_features_labels(parameters, edge_index_list, adjacency_matrices_source_information_list, node_features_list, edge_attr_list, arrowhead_solution_list):
    '''

    :param parameters: dictionary with parameters set in parameters.json file
    :param graph:
    :param node_features:
    :param arrowhead_solution:
    :return:
    '''

    np.save(os.path.join(parameters["output_directory"], parameters["dataset_name"] + "_X.npy"), np.array(node_features_list))
    np.save(os.path.join(parameters["output_directory"], parameters["dataset_name"] + "_edge_attr.npy"), np.array(edge_attr_list))
    np.save(os.path.join(parameters["output_directory"], parameters["dataset_name"] + "_edge_index.npy"), np.array(edge_index_list))
    np.save(os.path.join(parameters["output_directory"], parameters["dataset_name"] + "_source_information.npy"), np.array(adjacency_matrices_source_information_list))
    np.save(os.path.join(parameters["output_directory"], parameters["dataset_name"] + "_y.npy"), np.array(arrowhead_solution_list))