import sys
sys.path.insert(1, './tad_detection/')
sys.path.insert(1, './tad_detection/preprocessing/')
sys.path.insert(1, './tad_detection/model/')
sys.path.insert(1, './tad_detection/evaluation/')

import argparse
import os
from utils_general import load_parameters, set_up_logger
from utils_evaluate import load_predicted_tads_per_tad_prediction_methods, jaccard_index, venn_diagram_visualization, generate_chromosome_lists, tad_region_size_calculation

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_parameters_json", help=" to JSOn with parameters.", type=str, required=True)
    args = parser.parse_args()
    path_parameters_json = args.path_parameters_json
    #path_parameters_json = "./tad_detection/evaluation/parameters.json"

    parameters = load_parameters(path_parameters_json)
    os.makedirs(parameters["output_directory"], exist_ok=True)
    os.makedirs(os.path.join(parameters["output_directory"], "evaluation"), exist_ok=True)
    os.makedirs(os.path.join(parameters["output_directory"], "evaluation", "tad_regions"), exist_ok=True)
    os.makedirs(os.path.join(parameters["output_directory"], "evaluation", "venn_diagrams"), exist_ok=True)
    os.makedirs(os.path.join(parameters["output_directory"], "evaluation", "jaccard_indices"), exist_ok=True)
    logger = set_up_logger('evaluation', parameters)
    logger.debug('Start evaluation logger.')

    predicted_tads_per_tad_prediction_methods = load_predicted_tads_per_tad_prediction_methods(parameters)

    jaccard_index(parameters["tad_prediction_methods"], predicted_tads_per_tad_prediction_methods)

    venn_diagram_visualization(parameters, parameters["tad_prediction_methods"], predicted_tads_per_tad_prediction_methods)

    #genomic_annotations_histogram() #TODO

    parameters["chromsomes_int"], parameters["chromosomes_str_long"], parameters["chromosomes_str_short"] = generate_chromosome_lists(parameters)
    tad_region_size_calculation(parameters, predicted_tads_per_tad_prediction_methods)