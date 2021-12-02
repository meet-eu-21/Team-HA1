import matplotlib.pyplot as plt
import numpy as np

def hic_map_generation(parameters, adjacency_matrices_list, adjacency_matrices_source_information_list):

    for adjacency_matrix, source_information in zip(adjacency_matrices_list, adjacency_matrices_source_information_list):

        plt.imshow(adjacency_matrix, cmap="Reds")
        plt.title("HiC matrix of " + source_information)
        plt.xlabel("Genomic bins")
        plt.ylabel("Genomic bins")
        plt.gcf().set_size_inches(30, 30)
        #plt.show()
        plt.savefig(parameters["output_directory"], "preprocessing", "hic_contact_map_(heatmap)_of_ " + source_information + ".png")

def histogram_interaction_count_hic_map(parameters, adjacency_matrices_list, adjacency_matrices_source_information_list):
    for adjacency_matrix, source_information in zip(adjacency_matrices_list, adjacency_matrices_source_information_list):

        distribution_sum_row = []

        for row in adjacency_matrix:
            distribution_sum_row.append(sum(row))

        distribution_sum_row

        plt.title("Distribution interaction count sums for each row")
        plt.xlabel("interaction count sums")
        plt.ylabel("Prevalence of rows within bin with specific read count sum")
        plt.hist(distribution_sum_row, bins=50)
        #plt.show()
        plt.savefig(parameters["output_directory"], "preprocessing", "histogram_sums_interaction_values_in_rows_in_adjacency_matrix_ " + source_information + ".png")