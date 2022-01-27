import sys
sys.path.insert(1, './tad_detection/')
sys.path.insert(1, './tad_detection/preprocessing/')
sys.path.insert(1, './tad_detection/model/')
sys.path.insert(1, './tad_detection/evaluation/')

from utils_preprocessing import generate_chromosome_lists, chromosome_length_dict
from graph_generation import load_ccmap_file
import argparse
from utils_general import load_parameters


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create dictionary of chromosome lengths for usage in evaluation pipeline.')
    parser.add_argument("--path_parameters_json", help="path to JSON with parameters.", type=str, required=True)
    args = parser.parse_args()
    path_parameters_json = args.path_parameters_json
    # path_parameters_json = "./tad_detection/preprocessing/parameters.json"

    parameters = load_parameters(path_parameters_json)

    parameters["chromsomes_int"], parameters["chromosomes_str_long"], parameters["chromosomes_str_short"] = generate_chromosome_lists(parameters)

    adjacency_matrices_list, adjacency_matrices_source_information_list = load_ccmap_file(parameters)

    chr_len_dict = chromosome_length_dict(parameters, adjacency_matrices_list, adjacency_matrices_source_information_list)