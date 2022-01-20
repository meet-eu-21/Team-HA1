import sys
sys.path.insert(1, './tad_detection/')
sys.path.insert(1, './tad_detection/preprocessing/')
sys.path.insert(1, './tad_detection/model/')
sys.path.insert(1, './tad_detection/evaluation/')

import argparse
import os
import numpy as np
from utils_general import load_parameters, set_up_logger
from evaluation.utils_evaluate import bed_to_dict_file, generate_chromosome_lists, load_predicted_tad_dict_per_methods
from preprocessing.arrowhead_solution import load_arrowhead_solution_dict, load_arrowhead_solution

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_parameters_json", help=" to JSOn with parameters.", type=str, required=True)
    args = parser.parse_args()
    path_parameters_json = args.path_parameters_json

    parameters = load_parameters(path_parameters_json)
    parameters["chromosomes_int"], parameters["chromosomes_str_long"], parameters["chromosomes_str_short"] = generate_chromosome_lists(parameters)

    os.makedirs(parameters["output_directory"], exist_ok=True)
    os.makedirs(os.path.join(parameters["output_directory"], "evaluation"), exist_ok=True)
    os.makedirs(os.path.join(parameters["output_directory"], "evaluation", "tad_regions"), exist_ok=True)
    os.makedirs(os.path.join(parameters["output_directory"], "evaluation", "venn_diagrams"), exist_ok=True)
    os.makedirs(os.path.join(parameters["output_directory"], "evaluation", "jaccard_indices"), exist_ok=True)
    logger = set_up_logger('evaluation', parameters)
    logger.debug('Start evaluation logger.')

    bedfolder = '/Users/Charlotte/TopResults/100kb/GM12878/'
    chr_order_list = parameters["chromosomes_str_short"]
    del chr_order_list[-1]

    output_folder = "/Users/Charlotte/MeetEU/tad_detection/evaluation/results/"
    os.makedirs(output_folder, exist_ok=True)
    tad_dict = bed_to_dict_file(parameters["chromosomes_str_short"], bedfolder, chr_order_list, "TopDom", "GM12878", 100000, output_folder)

    # solution_dict_3 = load_arrowhead_solution_dict(parameters, 3)
    # print(solution_dict_3)
    #
    tad_dict_AH = load_arrowhead_solution_dict(parameters)
    # print(tad_dict_AH)

    # AH_npy = np.load("/Users/Charlotte/MeetEU/tad_detection/evaluation/results/AH_GM12878_100kb_v2.npy", allow_pickle=True)
    # TD_npy = np.load("/Users/Charlotte/MeetEU/tad_detection/evaluation/results/TopDomGM12878_100kb_v2.npy", allow_pickle=True)
    # print(AH_npy[1:10])
    # tad_dict = load_predicted_tad_dict_per_methods(parameters)
    # print(tad_dict)
    # print(len(tad_dict))