import matplotlib.pyplot as plt
import numpy as np
import os

def hic_map_generation(parameters, adjacency_matrices_list, adjacency_matrices_source_information_list):
    '''
    Function generates a Hi-C map for every adjacency matrix in adjacency_matrices_list.

    :param parameters: dictionary with parameters set in parameters.json file
    :param adjacency_matrices_list: adjacency matrices separated for chromosomes and cell line
    :param adjacency_matrices_source_information_list: source information for adjacency_matrices_list indicating the chromosome of the corresponding adjacency matrix
    '''

    for adjacency_matrix_cell_line, source_information_cell_line in zip(adjacency_matrices_list, adjacency_matrices_source_information_list):
        for adjacency_matrix, source_information in zip(adjacency_matrix_cell_line, source_information_cell_line):

            plt.imshow(adjacency_matrix, cmap="Reds")
            plt.title("HiC matrix of " + source_information)
            plt.xlabel("Genomic bins")
            plt.ylabel("Genomic bins")
            plt.gcf().set_size_inches(30, 30)
            #plt.show()
            plt.savefig(os.path.join(parameters["output_directory"], parameters["dataset_name"], "preprocessing", "hic_contact_map_(heatmap)_of_ " + source_information + ".png"))
            plt.close()

def histogram_interaction_count_hic_map(parameters, adjacency_matrices_list, adjacency_matrices_source_information_list):
    '''
    Function generates a histogram of the Hi-C distribution for every adjacency matrix in adjacency_matrices_list.

    :param parameters: dictionary with parameters set in parameters.json file
    :param adjacency_matrices_list: adjacency matrices separated for chromosomes and cell line
    :param adjacency_matrices_source_information_list: source information for adjacency_matrices_list indicating the chromosome of the corresponding adjacency matrix
    :return:
    '''

    for adjacency_matrix_cell_line, source_information_cell_line in zip(adjacency_matrices_list, adjacency_matrices_source_information_list):
        for adjacency_matrix, source_information in zip(adjacency_matrix_cell_line, source_information_cell_line):

            distribution_sum_row = []

            for row in adjacency_matrix:
                distribution_sum_row.append(sum(row))

            distribution_sum_row

            plt.title("Distribution interaction count sums for each row")
            plt.xlabel("interaction count sums")
            plt.ylabel("Prevalence of rows within bin with specific read count sum")
            plt.hist(distribution_sum_row, bins=50)
            #plt.show()
            plt.savefig(os.path.join(parameters["output_directory"], parameters["dataset_name"], "preprocessing", "histogram_sums_interaction_values_in_rows_in_adjacency_matrix_ " + source_information + ".png"))
            plt.close()