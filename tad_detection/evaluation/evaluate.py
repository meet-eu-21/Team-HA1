import sys
sys.path.insert(1, './tad_detection/')
sys.path.insert(1, './tad_detection/preprocessing/')
sys.path.insert(1, './tad_detection/model/')
sys.path.insert(1, './tad_detection/evaluation/')

import argparse
import os
from utils_general import load_parameters, set_up_logger
from evaluation.utils_evaluate import load_predicted_tad_dict_per_methods, \
    generate_chromosome_lists, tad_region_size_calculation, venn_diagram_visualization, \
    jaccard_index_from_tad_dict, flatten_tad_dict, hic_map_generation_evaluation

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run evaluation pipeline on multiple experiments and comparing results by calculating statistics (Size TADs, Count TADs, Jaccard index etc.), creating VEnn-Diagrams and Hi-C maps with TAD regions.')
    parser.add_argument("--path_parameters_json", help="path to JSON with parameters.", type=str, required=True)
    args = parser.parse_args()
    path_parameters_json = args.path_parameters_json

    parameters = load_parameters(path_parameters_json)
    parameters["chromosomes_int"], parameters["chromosomes_str_long"], parameters["chromosomes_str_short"] = generate_chromosome_lists(parameters)

    os.makedirs(parameters["output_directory"], exist_ok=True)
    os.makedirs(os.path.join(parameters["output_directory"], "evaluation"), exist_ok=True)
    os.makedirs(os.path.join(parameters["output_directory"], "evaluation", "tad_regions", "25kb"), exist_ok=True)
    os.makedirs(os.path.join(parameters["output_directory"], "evaluation", "tad_regions", "100kb"), exist_ok=True)
    os.makedirs(os.path.join(parameters["output_directory"], "evaluation", "venn_diagrams", "25kb"), exist_ok=True)
    os.makedirs(os.path.join(parameters["output_directory"], "evaluation", "venn_diagrams", "100kb"), exist_ok=True)
    os.makedirs(os.path.join(parameters["output_directory"], "evaluation", "jaccard_indices", "25kb"), exist_ok=True)
    os.makedirs(os.path.join(parameters["output_directory"], "evaluation", "jaccard_indices", "100kb"), exist_ok=True)

    logger = set_up_logger('evaluation', parameters)
    logger.debug('Start evaluation logger.')

    # predicted_tads_per_tad_prediction_methods = load_predicted_tads_per_tad_prediction_methods(parameters)
    # tad_dict = pred_tads_to_dict(parameters, predicted_tads_per_tad_prediction_methods)

    tad_dict = load_predicted_tad_dict_per_methods(parameters)

    flat_tad_dict = flatten_tad_dict(tad_dict)
    for chromosome in parameters["chromosomes_str_short"]:
        pred_tads_per_method_chrom = []
        for method in parameters["tad_prediction_methods"]:
            if not(chromosome in flat_tad_dict[method].keys()):
                continue
            pred_tads_per_method_chrom.append(flat_tad_dict[method][chromosome])
        if not(len(pred_tads_per_method_chrom)>1):
            continue

        if len(pred_tads_per_method_chrom) == len(parameters["tad_prediction_methods"]):
            venn_diagram_visualization(parameters, parameters["tad_prediction_methods"], pred_tads_per_method_chrom,
                                   chromosome)
            print(chromosome)

    jaccard_index_from_tad_dict(parameters, parameters["tad_prediction_methods"], flat_tad_dict)

    tad_region_size_calculation(parameters, tad_dict)

    hic_map_generation_evaluation(parameters, tad_dict)