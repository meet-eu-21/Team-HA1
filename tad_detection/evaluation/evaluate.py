import sys
sys.path.insert(1, './tad_detection/')
sys.path.insert(1, './tad_detection/preprocessing/')
sys.path.insert(1, './tad_detection/model/')
sys.path.insert(1, './tad_detection/evaluation/')

import argparse
import os
from utils_general import load_parameters, set_up_logger
from evaluation.utils_evaluate import load_predicted_tad_dict_per_methods, \
    generate_chromosome_lists, pred_tads_to_dict, tad_region_size_calculation, venn_diagram_visualization, \
    jaccard_index_from_tad_dict, flatten_tad_dict, create_adj_matrix
from gcMapExplorer import lib as gmlib
from tad_detection.preprocessing.visualization import hic_map_generation


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
        print(chromosome)
        venn_diagram_visualization(parameters, parameters["tad_prediction_methods"], pred_tads_per_method_chrom,
                                   chromosome)

    jaccard_index_from_tad_dict(parameters, parameters["tad_prediction_methods"], flat_tad_dict)

    tad_region_size_calculation(parameters, tad_dict)

    cell_line = parameters["cell_line"]
    for method in parameters["tad_prediction_methods"]:
        for chromosome in parameters["chromosomes_str_short"]:
            ccmap_path = "../../cmap_files/" + str(
                int(parameters["scaling_factor"]/1000)) + "kb/" + cell_line + "/intra/cmap_" + chromosome + ".ccmap"

            if parameters["scaling_factor"] == 100000:
                ccmap = gmlib.ccmap.load_ccmap(ccmap_path)
                ccmap.make_readable()
                chr_len = len(ccmap.matrix)
                chr_len = chr_len[0]
                chr_len = 3000
            else:
                ccmap = gmlib.ccmap.load_ccmap(ccmap_path)
                ccmap.make_readable()
                chr_len = ccmap.shape()
                chr_len = chr_len[0]
            print(chromosome)
            adj = create_adj_matrix(chr_len, tad_dict[method][chromosome])

            adjacency_matrices_source_information_list = [cell_line + "_" + method + "_" + str(int(parameters["scaling_factor"]/1000)) + "kb_chromosome_" + chromosome]
            hic_map_generation(parameters, [[adj]], [adjacency_matrices_source_information_list])