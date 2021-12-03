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

    path_list_dict = {}
    adjacency_matrices_list = []
    adjacency_matrices_source_information_list = []

    for cell_line in parameters["cell_lines"]:
        path_list_dict[cell_line] = {}
        for chromosome in parameters["chromosomes_str_long"]:
            adjacency_matrices_source_information_list.append(cell_line  + "-" + chromosome)
            path_list_dict[cell_line][chromosome]  = list(Path(os.path.join(parameters["hic_matrix_directory"], cell_line, "/" + parameters["resolution_hic_matrix_string"] + "_resolution_intrachromosomal/")).rglob(chromosome_name + "_100kb.RAWobserved"))

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

    with open(os.path.join(parameters["output_directory"], "preprocessing", 'dictionary_paths_of_used_hic_matrices_' + parameters["resolution_hic_matrix_string"] + '.pickle'), 'wb') as handle:
        pickle.dump(path_list_dict, handle, protocol=pickle.DEFAULT_PROTOCOL)

    logger.info("Wrote a dictionary with the paths of all used Hi-C matrices to: " + os.path.join(parameters["output_directory"], "preprocessing", 'dictionary_paths_of_used_hic_matrices_' + parameters["resolution_hic_matrix_string"] + '.pickle'))

    #TODO
    #GIBT ES DIESES DICTIONARY WIRKLICH? HABE ICH NOCH NRIGENDWO GESEHEN

    #TODO - Paul
    #ICH kann nicht erkennen, inwiefern die ccmap files hier auch gespeichert werden???

    return adjacency_matrices_list, adjacency_matrices_source_information_list

def load_ccmap_file(parameters):

    adjacency_matrices_list = []
    adjacency_matrices_list_cell_line = []

    adjacency_matrices_source_information_list = []
    adjacency_matrices_source_information_list_cell_line = []

    for cell_line in parameters["cell_lines"]:
        for chromosome in parameters["chromosomes_str_short"]:
            adjacency_matrices_source_information_list_cell_line.append(cell_line + "-" + chromosome)
            adjacency_matrix_chromosome = gmlib.ccmap.load_ccmap("./cmap_files/intra/cmap_" + chromosome + ".ccmap")
            adjacency_matrix_chromosome.make_readable()
            adjacency_matrices_list_cell_line.append(np.array(adjacency_matrix_chromosome.matrix))

        adjacency_matrices_list.append(adjacency_matrices_list_cell_line)
        adjacency_matrices_source_information_list.append(adjacency_matrices_source_information_list_cell_line)

    return adjacency_matrices_list, adjacency_matrices_source_information_list

def statistics_adjacency_matrix(parameters, adjacency_matrices_list, adjacency_matrices_source_information_list):

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
            plt.savefig(os.path.join(parameters["output_directory"], "preprocessing", "histogram_interaction_values_in_adjacency_matrix_ " + source_information + ".png"))

def graph_filtering(parameters, adjacency_matrices_list):

    #TODO
    #Add logger statistics here.
    adjacency_matrices_list_filtered = adjacency_matrices_list

    if parameters["threshold_graph_vertex_filtering"] != "None":
        adjacency_matrices_list_filtered = []
        adjacency_matrices_cell_line_filtered = []
        for adjacency_matrices_cell_line in adjacency_matrices_list:
            for adjacency_matrix in adjacency_matrices_cell_line:
                ###
                genomic_bins = adjacency_matrix.shape[0]
                genomic_bins_delete = set()
                for genomic_bin in range(0, genomic_bins-1):
                    if len(adjacency_matrix[adjacency_matrix[genomic_bin] < parameters["threshold_graph_vertex_filtering"]]) < parameters["graph_vertex_filtering_min_val"]:
                        genomic_bins_delete.add(genomic_bin)
                adjacency_matrix = adjacency_matrix[list(set(range(0, genomic_bins-1)) - set(genomic_bins_delete)), :]
                adjacency_matrix = adjacency_matrix[:, list(set(range(0, genomic_bins-1)) - set(genomic_bins_delete))]
                adjacency_matrices_cell_line_filtered.append(adjacency_matrix)
            adjacency_matrices_list_filtered.append(adjacency_matrices_cell_line_filtered)
    if parameters["threshold_graph_edge_filtering"] != "None":
        adjacency_matrices_list_filtered = []
        adjacency_matrices_cell_line_filtered = []
        for adjacency_matrices_cell_line in adjacency_matrices_list:
            for adjacency_matrix in adjacency_matrices_cell_line:
                for i, j in zip(range(0, adjacency_matrix.shape[0]-1), range(0, adjacency_matrix.shape[1]-1)):
                    if adjacency_matrix[i,j] < parameters["threshold_graph_edge_filtering"]:
                        adjacency_matrix[i,j] = 0

                adjacency_matrices_cell_line_filtered.append(adjacency_matrix)
            adjacency_matrices_list_filtered.append(adjacency_matrices_cell_line_filtered)

    return adjacency_matrices_list_filtered
