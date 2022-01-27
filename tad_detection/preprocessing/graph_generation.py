import sys
sys.path.insert(1, './tad_detection/')
sys.path.insert(1, './tad_detection/preprocessing/')
sys.path.insert(1, './tad_detection/model/')
sys.path.insert(1, './tad_detection/evaluation/')

import logging
logger = logging.getLogger('preprocessing')

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import os
import pickle
import matplotlib.pyplot as plt
from gcMapExplorer import lib as gmlib

def load_adjacency_matrix(parameters):
    '''
    Function genertates adjacency_matrices_list by loading each adjacency matrix separately from raw data.

    :param parameters: dictionary with parameters set in parameters.json file
    :return adjacency_matrices_list: adjacency matrices separated for chromosomes and cell line
    :return adjacency_matrices_source_information_list: source information for adjacency_matrices_list indicating the chromosome of the corresponding adjacency matrix
    '''

    path_list_dict = {}
    adjacency_matrices_list = []
    adjacency_matrices_source_information_list = []

    for cell_line in parameters["cell_lines"]:
        path_list_dict[cell_line] = {}
        for chromosome in parameters["chromosomes_str_long"]:
            adjacency_matrices_source_information_list.append(cell_line  + "-" + chromosome)
            path_list_dict[cell_line][chromosome] = list(Path(os.path.join(parameters["hic_matrix_directory"], cell_line, parameters["resolution_hic_matrix_string"] + "_resolution_intrachromosomal/")).rglob(chromosome + "_"  + parameters["resolution_hic_matrix_string"] + ".RAWobserved"))

            lines = []

            for file in path_list_dict[cell_line][chromosome]:
                with open(file, 'r') as f:
                    lines_sub = f.read().splitlines()
                print(len(lines_sub))
                lines = lines + lines_sub
                f.close()

            data = [i.split('\t') for i in lines]
            z = list(zip(*data))

            row_indices = np.array(list(map(int, z[0])))
            row_indices = row_indices / parameters["scaling_factor"]
            row_indices = row_indices.astype(int)

            column_indices = np.array(list(map(int, z[1])))
            column_indices = column_indices / parameters["scaling_factor"]
            column_indices = column_indices.astype(int)

            values = list(map(float, z[2]))

            m = max(row_indices) + 1
            n = max(column_indices) + 1
            p = max([m, n])

            adjacency_matrix = np.zeros((p, p))

            adjacency_matrix[row_indices, column_indices] = values

            adjacency_matrices_list.append(adjacency_matrix)

    with open(os.path.join(parameters["output_directory"], "preprocessing", 'dictionary_paths_of_used_hic_matrices_' + parameters["resolution_hic_matrix_string"] + '.pickle'), 'wb') as handle:
        pickle.dump(path_list_dict, handle, protocol=pickle.DEFAULT_PROTOCOL)

    logger.info("Wrote a dictionary with the paths of all used Hi-C matrices to: " + os.path.join(parameters["output_directory"], "preprocessing", 'dictionary_paths_of_used_hic_matrices_' + parameters["resolution_hic_matrix_string"] + '.pickle'))

    return adjacency_matrices_list, adjacency_matrices_source_information_list

def load_ccmap_file(parameters):
    '''

    :param parameters: dictionary with parameters set in parameters.json file
    :return adjacency_matrices_list: adjacency matrices separated for chromosomes and cell line
    :return adjacency_matrices_source_information_list: source information for adjacency_matrices_list indicating the chromosome of the corresponding adjacency matrix
    '''

    adjacency_matrices_list = []
    adjacency_matrices_source_information_list = []

    for cell_line in parameters["cell_lines"]:
        adjacency_matrices_list_cell_line = []
        adjacency_matrices_source_information_list_cell_line = []
        for chromosome in parameters["chromosomes_str_short"]:
            logger.info(f"Importing chromosome {chromosome}")
            adjacency_matrices_source_information_list_cell_line.append(cell_line + "-" + chromosome)
            adjacency_matrix_chromosome = gmlib.ccmap.load_ccmap(f'./cmap_files/{parameters["resolution_hic_matrix_string"]}/{cell_line}/intra/cmap_{chromosome}.ccmap')
            adjacency_matrix_chromosome.make_readable()
            adjacency_matrices_list_cell_line.append(np.array(adjacency_matrix_chromosome.matrix))

        adjacency_matrices_list.append(adjacency_matrices_list_cell_line)
        adjacency_matrices_source_information_list.append(adjacency_matrices_source_information_list_cell_line)

    return adjacency_matrices_list, adjacency_matrices_source_information_list

def statistics_adjacency_matrix(parameters, adjacency_matrices_list, adjacency_matrices_source_information_list):
    '''
    Function generates statistics for each adjacency matrix in adjacency_matrices_list including the diagonal sum and the distribution of interaction values. Also, it generates a histogram with the distribution of interaction values for each adjacency matrix.

    :param parameters: dictionary with parameters set in parameters.json file
    :param adjacency_matrices_list: adjacency matrices separated for chromosomes and cell line
    :param adjacency_matrices_source_information_list: source information for adjacency_matrices_list indicating the chromosome of the corresponding adjacency matrix
    '''

    for adjacency_matrix_cell_line, source_information_cell_line in zip(adjacency_matrices_list, adjacency_matrices_source_information_list):

        for adjacency_matrix, source_information in zip(adjacency_matrix_cell_line, source_information_cell_line):

            logger.info("Sum of diagonal of adjacency matrix from " + source_information + ": " + str(np.trace(adjacency_matrix)))
            #np.diagonal(adjacency_matrix)
            #Values diagonal

            logger.info("Quantiles of interaction values in adjacency matrix from " + source_information + ":")

            adjacency_matrix_flatten = adjacency_matrix.flatten()
            adjacency_matrix_flatten = np.array(adjacency_matrix_flatten)
            adjacency_matrix_flatten = adjacency_matrix_flatten[adjacency_matrix_flatten != 0]
            #adjacency_matrix_flatten

            logger.info("Minimum interaction value: " + str(np.quantile(adjacency_matrix_flatten, 0)))
            logger.info("0.1 quantile interaction value: " + str(np.quantile(adjacency_matrix_flatten, 0.1)))
            logger.info("0.2 quantile interaction value: " + str(np.quantile(adjacency_matrix_flatten, 0.2)))
            logger.info("0.3 quantile interaction value: " + str(np.quantile(adjacency_matrix_flatten, 0.3)))
            logger.info("0.4 quantile interaction value: " + str(np.quantile(adjacency_matrix_flatten, 0.4)))
            logger.info("0.5 quantile interaction value: " + str(np.quantile(adjacency_matrix_flatten, 0.5)))
            logger.info("0.6 quantile interaction value: " + str(np.quantile(adjacency_matrix_flatten, 0.6)))
            logger.info("0.7 quantile interaction value: " + str(np.quantile(adjacency_matrix_flatten, 0.7)))
            logger.info("0.8 quantile interaction value: " + str(np.quantile(adjacency_matrix_flatten, 0.8)))
            logger.info("0.9 quantile interaction value: " + str(np.quantile(adjacency_matrix_flatten, 0.9)))
            logger.info("Maximum interaction value: " + str(np.quantile(adjacency_matrix_flatten, 1.0)))

            plt.hist(adjacency_matrix_flatten, bins=10000)

            plt.title("Distribution interaction values in Hi-C matrix")
            plt.xlabel("Interaction count")
            plt.ylabel("Prevalence of values within bin with specific interaction count")

            plt.gcf().set_size_inches(18.5, 18.5)
            #plt.show()
            plt.savefig(os.path.join(parameters["output_directory"], parameters["dataset_name"], "preprocessing", "histogram_interaction_values_in_adjacency_matrix_ " + source_information + ".png"))
            plt.close()

def graph_filtering(parameters, adjacency_matrices_list, adjacency_matrices_source_information_list):
    '''
    Function filters vertices and edges by thresholds in parameters for each adjacency matrix in adjacency_matrices_list.

    :param parameters: dictionary with parameters set in parameters.json file
    :param adjacency_matrices_list: adjacency matrices separated for chromosomes and cell line
    :param adjacency_matrices_source_information_list: source information for adjacency_matrices_list indicating the chromosome of the corresponding adjacency matrix
    :return adjacency_matrices_list_filtered: filtered adjacency matrices separated for chromosomes and cell line
    '''

    adjacency_matrices_list_filtered = adjacency_matrices_list

    if parameters["threshold_graph_vertex_filtering"] != "None":
        adjacency_matrices_list_filtered = []
        adjacency_matrices_cell_line_filtered = []
        for adjacency_matrices_cell_line, adjacency_matrices_source_information_list_cell_line in zip(adjacency_matrices_list, adjacency_matrices_source_information_list):
            logger.info(f"Genomic bins graph filtering of chromosomes of cell line {adjacency_matrices_source_information_list_cell_line[0].split('-')[0]}")
            for adjacency_matrix, adjacency_matrices_source_information in zip(adjacency_matrices_cell_line, adjacency_matrices_source_information_list_cell_line):
                logger.info(f"Graph filtering of chromosome {adjacency_matrices_source_information.split('-')[1]} with {len(adjacency_matrix)} genomic bins before filtering.")
                genomic_bins = adjacency_matrix.shape[0]
                genomic_bins_delete = set()
                for genomic_bin in range(0, genomic_bins-1):
                    if len(adjacency_matrix[adjacency_matrix[genomic_bin] < parameters["threshold_graph_vertex_filtering"]]) > parameters["graph_vertex_filtering_min_val"]:
                        genomic_bins_delete.add(genomic_bin)
                adjacency_matrix = adjacency_matrix[list(set(range(0, genomic_bins-1)) - set(genomic_bins_delete)), :]
                adjacency_matrix = adjacency_matrix[:, list(set(range(0, genomic_bins-1)) - set(genomic_bins_delete))]
                adjacency_matrices_cell_line_filtered.append(adjacency_matrix)
                logger.info(f"Graph filtering of chromosome {adjacency_matrices_source_information.split('-')[1]} with {len(adjacency_matrix)} genomic bins after filtering.")
            adjacency_matrices_list_filtered.append(adjacency_matrices_cell_line_filtered)
    if parameters["threshold_graph_edge_filtering"] != "None":
        adjacency_matrices_list_filtered = []
        adjacency_matrices_cell_line_filtered = []
        for adjacency_matrices_cell_line, adjacency_matrices_source_information_list_cell_line in zip(adjacency_matrices_list, adjacency_matrices_source_information_list):
            logger.info(f"Vertices graph filtering of chromosomes of cell line {adjacency_matrices_source_information_list_cell_line[0].split('-')[0]}")
            for adjacency_matrix, adjacency_matrices_source_information in zip(adjacency_matrices_cell_line, adjacency_matrices_source_information_list_cell_line):
                count_vertices_before_filtering = (adjacency_matrix > 0).sum()
                logger.info(f"Graph filtering of chromosome {adjacency_matrices_source_information.split('-')[1]} with {count_vertices_before_filtering} vertices > 0 (Total amount of vertices including 0: {len(adjacency_matrix)*len(adjacency_matrix)}.) before filtering.")
                adjacency_matrix[adjacency_matrix < parameters["threshold_graph_edge_filtering"]] = 0
                count_vertices_after_filtering = (adjacency_matrix > 0).sum()
                logger.info(f"Graph filtering of chromosome {adjacency_matrices_source_information.split('-')[1]} with {count_vertices_after_filtering} vertices > 0 (Filtered {count_vertices_before_filtering-count_vertices_after_filtering} vertices in total.) after filtering.")

                adjacency_matrices_cell_line_filtered.append(adjacency_matrix)
            adjacency_matrices_list_filtered.append(adjacency_matrices_cell_line_filtered)

    return adjacency_matrices_list_filtered


def restrict_labels_solution_list(labels_solution_list, edge_index_list):
    '''
    Function cuts off labels in labels_solution_list, if edge_index in edge_index_list is shorter than labels. This is the case when the last genomic bins in the adjacency matrix do not contain any vertices.

    :param labels_solution_list: true labels separated for chromosomes and cell line
    :param edge_index_list: edge index representation of adjacency_matrices_list separated for chromosomes and cell line, reports the nodes and the edges between the nodes present in a graph
    :return labels_solution_list: true labels separated for chromosomes and cell line restricted by edge_index_list
    '''

    for cell_line_index, (labels_solution_list_cell_line, edge_index_list_cell_line) in enumerate(
            zip(labels_solution_list, edge_index_list)):
        for chromosome_index, (labels_solution_list_chromosome, edge_index_list_chromsome) in enumerate(
                zip(labels_solution_list_cell_line, edge_index_list_cell_line)):
            labels_solution_list[cell_line_index][chromosome_index] = labels_solution_list_chromosome[:max(set(edge_index_list_chromsome[0]) | set(edge_index_list_chromsome[1]))+1]

    return labels_solution_list

def restrict_node_features_list(node_feature_list, edge_index_list):
    '''
    Function cuts off node_features in node_feature_list, if edge_index in edge_index_list is shorter than node_features. This is the case when the last genomic bins in the adjacency matrix do not contain any vertices.

    :param node_feature_list: features of nodes in annotation matrices separated for chromosomes and cell line
    :param edge_index_list: edge index representation of adjacency_matrices_list separated for chromosomes and cell line, reports the nodes and the edges between the nodes present in a graph
    :return node_feature_list: features of nodes in annotation matrices separated for chromosomes and cell line restricted by edge_index_list
    '''

    for cell_line_index, (node_feature_list_cell_line, edge_index_list_cell_line) in enumerate(
            zip(node_feature_list, edge_index_list)):
        for chromosome_index, (node_feature_list_chromosome, edge_index_list_chromsome) in enumerate(
                zip(node_feature_list_cell_line, edge_index_list_cell_line)):
            node_feature_list[cell_line_index][chromosome_index] = node_feature_list_chromosome[:max(set(edge_index_list_chromsome[0]) | set(edge_index_list_chromsome[1]))+1]

    return node_feature_list