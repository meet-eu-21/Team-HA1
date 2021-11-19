import argparse
import os
from utils import load_parameters, set_up_logger, load_predicted_tads_per_tad_prediction_methods, jaccard_index, venn_diagram_visualization

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_parameters_json", help=" to JSOn with parameters.", type=str, required=True)
    args = parser.parse_args()
    path_parameters_json = args.path_parameters_json

    parameters = load_parameters(path_parameters_json)
    os.makedirs(parameters["output_path"], exist_ok=True)
    os.makedirs(os.path.join(parameters["output_path"], "evaluation"), exist_ok=True)
    os.makedirs(os.path.join(parameters["output_path"], "evaluation", "venn_diagrams"), exist_ok=True)
    os.makedirs(os.path.join(parameters["output_path"], "evaluation", "jaccard_indices"), exist_ok=True)
    set_up_logger(parameters)

    predicted_tads_per_tad_prediction_methods = load_predicted_tads_per_tad_prediction_methods(parameters)

    jaccard_index(parameters["tad_prediction_methods"], predicted_tads_per_tad_prediction_methods)

    venn_diagram_visualization(parameters, parameters["tad_prediction_methods"], predicted_tads_per_tad_prediction_methods)