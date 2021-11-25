from utils import load_parameters, set_up_logger, generate_chromosome_lists, extract_bins, save_adjacency_matrix_node_features_labels
from graph_generation import load_adjacency_matrix, statistics_adjacency_matrix, graph_filtering
from visualization import hic_map_generation, histogram_interaction_count_hic_map
from arrowhead_solution import load_arrowhead_solution
from node_annotations import load_genomic_annotations, load_housekeeping_genes, combine_genomic_annotations_and_housekeeping_genes
import os
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_parameters_json", help=" to JSON with parameters.", type=str, required=True)
    args = parser.parse_args()
    path_parameters_json = args.path_parameters_json

    parameters = load_parameters(path_parameters_json)
    os.makedirs(parameters["output_path"], exist_ok=True)
    os.makedirs(os.path.join(parameters["output_path"], "preprocessing"), exist_ok=True)
    set_up_logger(parameters)

    parameters["chromsomes_int"], parameters["chromosomes_str_long"], parameters["chromosomes_str_short"] = generate_chromosome_lists(parameters)

    adjacency_matrices_list, adjacency_matrices_source_information_list = load_adjacency_matrix(parameters)
    statistics_adjacency_matrix(adjacency_matrices_list, adjacency_matrices_source_information_list)
    adjacency_matrices_list = graph_filtering(parameters, adjacency_matrices_list)

    hic_map_generation(parameters, adjacency_matrices_list, adjacency_matrices_source_information_list)
    histogram_interaction_count_hic_map(parameters, adjacency_matrices_list, adjacency_matrices_source_information_list)

    arrowhead_solution = load_arrowhead_solution(parameters)
    genomic_annotations = load_genomic_annotations(parameters)
    housekeeping_genes = load_housekeeping_genes(parameters)

    #SOMEHOW ASSIGN TO BINS
    extract_bins()

    node_features = combine_genomic_annotations_and_housekeeping_genes(genomic_annotations, housekeeping_genes)

    save_adjacency_matrix_node_features_labels(parameters, adjacency_matrix, node_features, arrowhead_solution)