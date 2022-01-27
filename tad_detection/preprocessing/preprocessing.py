import sys
sys.path.insert(1, './tad_detection/')
sys.path.insert(1, './tad_detection/preprocessing/')
sys.path.insert(1, './tad_detection/model/')
sys.path.insert(1, './tad_detection/evaluation/')

from utils_general import load_parameters, dump_parameters, set_up_logger
from utils_preprocessing import generate_chromosome_lists, save_adjacency_matrix_node_features_labels, generate_edge_index_edge_attr_from_adjacency_matrix
from graph_generation import load_ccmap_file, statistics_adjacency_matrix, graph_filtering, restrict_labels_solution_list, restrict_node_features_list
from visualization import hic_map_generation, histogram_interaction_count_hic_map
from arrowhead_solution import load_arrowhead_solution, load_french_team_labels, one_hot_encode_labels_solution
from node_annotations import load_dict_genomic_annotations, load_dict_housekeeping_genes, combine_genomic_annotations_and_housekeeping_genes
import os
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create dataset consisting out of edge_index, edge_attr, source_information and labels from cmap files, genomic annotations and housekeeping dicts and arrowhead solution or the labels used by the french team.')
    parser.add_argument("--path_parameters_json", help="path to JSON with parameters.", type=str, required=True)
    args = parser.parse_args()
    path_parameters_json = args.path_parameters_json
    # path_parameters_json = "./tad_detection/preprocessing/parameters.json"

    parameters = load_parameters(path_parameters_json)
    os.makedirs(parameters["output_directory"], exist_ok=True)
    os.makedirs(os.path.join(parameters["output_directory"], parameters["dataset_name"]), exist_ok=True)
    os.makedirs(os.path.join(parameters["output_directory"], parameters["dataset_name"], "preprocessing"), exist_ok=True)
    dump_parameters(parameters)

    logger = set_up_logger('preprocessing', parameters)
    logger.debug('Start preprocessing logger.')

    parameters["chromsomes_int"], parameters["chromosomes_str_long"], parameters["chromosomes_str_short"] = generate_chromosome_lists(parameters)

    adjacency_matrices_list, adjacency_matrices_source_information_list = load_ccmap_file(parameters)

    if parameters["generate_plots_statistics"] == "True":
        statistics_adjacency_matrix(parameters, adjacency_matrices_list, adjacency_matrices_source_information_list)

    adjacency_matrices_list = graph_filtering(parameters, adjacency_matrices_list, adjacency_matrices_source_information_list)
    edge_index_list, edge_attr_list = generate_edge_index_edge_attr_from_adjacency_matrix(parameters, adjacency_matrices_list)

    if parameters["generate_plots_statistics"] == "True":
        hic_map_generation(parameters, adjacency_matrices_list, adjacency_matrices_source_information_list)
        histogram_interaction_count_hic_map(parameters, adjacency_matrices_list, adjacency_matrices_source_information_list)

    if parameters["label_types"] == "arrowhead":
        _, labels_solution_list = load_arrowhead_solution(parameters)
    elif parameters["label_types"] == "french_team_labels":
        labels_solution_list = load_french_team_labels(parameters, adjacency_matrices_list, adjacency_matrices_source_information_list)
    else:
        raise NotImplementedError("Please provide 'arrowhead' or 'french_team_labels' for the variable 'label_types' in parameters.json.")

    labels_solution_list = one_hot_encode_labels_solution(parameters, adjacency_matrices_list, labels_solution_list)

    dict_genomic_annotations = {}
    for cell_line in parameters["cell_lines"]:
        dict_genomic_annotations[cell_line] = load_dict_genomic_annotations(parameters, cell_line)

    dict_housekeeping_genes = load_dict_housekeeping_genes(parameters)

    node_features_list = combine_genomic_annotations_and_housekeeping_genes(parameters, labels_solution_list, adjacency_matrices_source_information_list, dict_genomic_annotations, dict_housekeeping_genes)

    node_features_list = restrict_node_features_list(node_features_list, edge_index_list)
    labels_solution_list = restrict_labels_solution_list(labels_solution_list, edge_index_list)

    save_adjacency_matrix_node_features_labels(parameters, edge_index_list, adjacency_matrices_source_information_list, node_features_list, edge_attr_list, labels_solution_list)
    logger.info(f"Wrote data to {parameters['output_directory']}")