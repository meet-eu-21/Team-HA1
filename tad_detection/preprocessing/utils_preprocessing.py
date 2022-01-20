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
import pickle

def generate_chromosome_lists(parameters):
    '''
    Function generates lists of chromosomes in different representations (integers and strings) based on the input in the parameters.json file for the usage in other functions.

    :param parameters: dictionary with parameters set in parameters.json file
    :return chromosomes_int: integer list of chromosomes
    :return chromosomes_str_long: string list of chromosomes in the format "chr<chromosome>"
    :return chromosomes_str_short: string list of chromosomes in the format "<chromosome>"
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
    '''
    Function generates the edge_index and edge_attr matrices from an adjacency matrix.

    :param parameters: dictionary with parameters set in parameters.json file
    :param adjacency_matrices_list: adjacency matrices separated for chromosomes and cell line
    :return edge_index: edge index representation of adjacency_matrices_list separated for chromosomes and cell line, reports the nodes and the edges between the nodes present in a graph
    :return edge_attr: edge attributes representation of adjacency_matrices_list separated for chromosomes and cell line, reports the attributes, e.g. the weight, of each edge in edge_index
    '''

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
    Function saves edge_index_list, adjacency_matrices_source_information_list, node_features_list, edge_attr_list and arrowhead_solution_list to the output directory given in parameters.

    :param parameters: dictionary with parameters set in parameters.json file
    :param edge_index_list: edge index representation of adjacency_matrices_list separated for chromosomes and cell line, reports the nodes and the edges between the nodes present in a graph
    :param adjacency_matrices_source_information_list: source information for adjacency_matrices_list indicating the chromosome of the corresponding adjacency matrix
    :param node_features_list: features of nodes given in edge_index_list separated for chromosomes and cell line
    :param edge_attr_list: edge attributes representation of adjacency_matrices_list separated for chromosomes and cell line, reports the attributes, e.g. the weight, of each edge in edge_index
    :param arrowhead_solution_list: TAD classification for genomic bins in original HiC-maps/ graph nodes called by Arrowhead method (Ground truth) separated for chromosomes and cell line
    '''

    np.save(os.path.join(parameters["output_directory"], parameters["dataset_name"], parameters["dataset_name"] + "_X.npy"), np.array(node_features_list))
    np.save(os.path.join(parameters["output_directory"], parameters["dataset_name"], parameters["dataset_name"] + "_edge_attr.npy"), np.array(edge_attr_list))
    np.save(os.path.join(parameters["output_directory"], parameters["dataset_name"], parameters["dataset_name"] + "_edge_index.npy"), np.array(edge_index_list))
    np.save(os.path.join(parameters["output_directory"], parameters["dataset_name"], parameters["dataset_name"] + "_source_information.npy"), np.array(adjacency_matrices_source_information_list))
    np.save(os.path.join(parameters["output_directory"], parameters["dataset_name"], parameters["dataset_name"] + "_y.npy"), np.array(arrowhead_solution_list))

def chromosome_length_dict(parameters, adjacency_matrices_list, adjacency_matrices_source_information_list):
    '''
    Function generates a dictionary with chromosome lengths.

    :param parameters: dictionary with parameters set in parameters.json file
    :param adjacency_matrices_list: adjacency matrices separated for chromosomes and cell line
    :param adjacency_matrices_source_information_list: source information for adjacency_matrices_list indicating the chromosome of the corresponding adjacency matrix
    '''

    chr_len_dict = dict.fromkeys(parameters["cell_lines"], {})
    for adjacency_matrices_cell_line, source_information_cell_line in zip(adjacency_matrices_list, adjacency_matrices_source_information_list):
        for adjacency_matrix, source_information in zip(adjacency_matrices_cell_line, source_information_cell_line):
            chr_len_dict["-".join(source_information.split("-")[0:-1])][source_information.split("-")[-1]] = adjacency_matrix.shape[0]

    output_path = os.path.join("./ressources/" + str(int(parameters["scaling_factor"]/1000)) + "kb_chr_len_dict.pickle")
    with open(output_path, 'wb') as handle:
        pickle.dump(chr_len_dict, handle, protocol=pickle.DEFAULT_PROTOCOL)

    return chr_len_dict