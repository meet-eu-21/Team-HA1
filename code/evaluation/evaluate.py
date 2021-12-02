import sys
sys.path.insert(1, './preprocessing/')
sys.path.insert(1, './model/')
sys.path.insert(1, './evaluation/')

import argparse
import os
from utils_evaluate import load_parameters, set_up_logger, load_predicted_tads_per_tad_prediction_methods, jaccard_index, venn_diagram_visualization

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_parameters_json", help=" to JSOn with parameters.", type=str, required=True)
    args = parser.parse_args()
    path_parameters_json = args.path_parameters_json

    parameters = load_parameters(path_parameters_json)
    os.makedirs(parameters["output_directory"], exist_ok=True)
    os.makedirs(os.path.join(parameters["output_directory"], "evaluation"), exist_ok=True)
    os.makedirs(os.path.join(parameters["output_directory"], "evaluation", "venn_diagrams"), exist_ok=True)
    os.makedirs(os.path.join(parameters["output_directory"], "evaluation", "jaccard_indices"), exist_ok=True)
    set_up_logger(parameters)

    predicted_tads_per_tad_prediction_methods = load_predicted_tads_per_tad_prediction_methods(parameters)

    jaccard_index(parameters["tad_prediction_methods"], predicted_tads_per_tad_prediction_methods)

    venn_diagram_visualization(parameters, parameters["tad_prediction_methods"], predicted_tads_per_tad_prediction_methods)