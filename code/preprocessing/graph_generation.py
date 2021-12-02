import sys
sys.path.insert(1, './preprocessing/')
sys.path.insert(1, './model/')
sys.path.insert(1, './evaluation/')

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import os
import pickle
import matplotlib.pyplot as plt

def load_adjacency_matrix(parameters):

    path_list_dict = {}
    adjacency_matrices_list = []
    adjacency_matrices_source_information_list = []

    for cell_line in parameters["cell_lines"]:
        path_list_dict[cell_line] = {}
        for chromosome in parameters["chromosomes_str_long"]:
            adjacency_matrices_source_information_list.append(cell_line  + "-" + chromosome)
            path_list_dict[cell_line][chromosome]  = list(Path(os.path.join(parameters["hic_matrix_directory"], cell_line, "/100kb_resolution_intrachromosomal/")).rglob(chromosome_name + "_100kb.RAWobserved"))

            lines = []

            for file in path_list_dict[chromosome]:
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

    with open(os.path.join(parameters["output_directory"], "preprocessing", 'dictionary_paths_of_used_hic_matrices.pickle'), 'wb') as handle:
        pickle.dump(path_list_dict, handle, protocol=pickle.DEFAULT_PROTOCOL)

    logger.info("Wrote a dictionary with the paths of all used Hi-C amtrices to: " + os.path.join(parameters["output_directory"], "preprocessing", 'dictionary_paths_of_used_hic_matrices.pickle'))

    return adjacency_matrices_list, adjacency_matrices_source_information_list

def statistics_adjacency_matrix(parameters, adjacency_matrices_list, adjacency_matrices_source_information_list):

    for adjacency_matrix, source_information in zip(adjacency_matrices_list, adjacency_matrices_source_information_list):

        logger.info("Sum of diagonal of adjacency matrix from " + source_information + ": " + str(np.trace(adjacency_matrix)))
        #np.diagonal(adjacency_matrix)
        #Values diagonal

        logger.info("Quantiles of interaction values in adjacency matrix from " + source_information + ":")

        adjacency_matrix_flatten = adjacency_matrix.flatten()
        adjacency_matrix_flatten = np.array(adjacency_matrix_flatten)
        adjacency_matrix_flatten = adjacency_matrix_flatten[adjacency_matrix_flatten != 0]
        adjacency_matrix_flatten

        logger.info("Minimum interaction value: " + np.quantile(adjacency_matrix_flatten, 0))
        logger.info("0.1 quantile interaction value: " + np.quantile(adjacency_matrix_flatten, 0.1))
        logger.info("0.2 quantile interaction value: " + np.quantile(adjacency_matrix_flatten, 0.2))
        logger.info("0.3 quantile interaction value: " + np.quantile(adjacency_matrix_flatten, 0.3))
        logger.info("0.4 quantile interaction value: " + np.quantile(adjacency_matrix_flatten, 0.4))
        logger.info("0.5 quantile interaction value: " + np.quantile(adjacency_matrix_flatten, 0.5))
        logger.info("0.6 quantile interaction value: " + np.quantile(adjacency_matrix_flatten, 0.6))
        logger.info("0.7 quantile interaction value: " + np.quantile(adjacency_matrix_flatten, 0.7))
        logger.info("0.8 quantile interaction value: " + np.quantile(adjacency_matrix_flatten, 0.8))
        logger.info("0.9 quantile interaction value: " + np.quantile(adjacency_matrix_flatten, 0.9))
        logger.info("Maximum interaction value: " + np.quantile(adjacency_matrix_flatten, 1.0))


        plt.hist(adjacency_matrix_flatten, bins=10000)

        plt.title("Distribution interaction values in Hi-C matrix")
        plt.xlabel("Interaction count")
        plt.ylabel("Prevalence of values within bin with specific interaction count")

        plt.gcf().set_size_inches(18.5, 18.5)
        #plt.show()
        plt.savefig(parameters["output_directory"], "preprocessing", "histogram_interaction_values_in_adjacency_matrix_ " + source_information + ".png")








def graph_filtering(parameters, adjacency_matrices_list):

    if parameters["threshold_graph_vertex_filtering"] != "None":
        for adjacency_matrix in adjacency_matrices_list:
            ###
            genomic_bins = adjacency_matrix.shape[0]
            genomic_bins_delete = set()
            for genomic_bin in genomic_bins:
                if len(adjacency_matrix[adjacency_matrix[genomic_bin] < parameters["threshold_graph_vertex_filtering"]]): #GROESER/ KLEINER DEFINIEREN, THEORETISCH KOENNTEN AUCH NEUE AUFTRETEN
                    adjacency_matrix = adjacency_matrix[set(range(0, genomic_bins)) - set(genomic_bins_delete), set(range(0, genomic_bins)) - set(genomic_bins_delete)]
    if parameters["threshold_graph_edge_filtering"] != "None":
        for adjacency_matrix in adjacency_matrices_list:
            adjacency_matrix[adjacency_matrix > parameters["threshold_graph_edge_filtering"]] = 0 #GROESER/ KLEINER DEFINIEREN

    return adjacency_matrices_list
