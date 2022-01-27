import sys
sys.path.insert(1, './tad_detection/')
sys.path.insert(1, './tad_detection/preprocessing/')
sys.path.insert(1, './tad_detection/model/')
sys.path.insert(1, './tad_detection/evaluation/')

import pandas as pd
import os
import numpy as np
import argparse
from tad_detection.utils_general import load_parameters, set_up_logger
from tad_detection.model.utils_model import load_data, save_tad_list
from tad_detection.evaluation.utils_evaluate import generate_chromosome_lists
import pickle


def load_arrowhead_solution(parameters, isdict=False):
    '''
    Function loads dataframes with TAD classification called by Arrowhead method (Ground truth) and extracts the TADs in original HiC-maps/ graph nodes and the TAD classification for genomic bins in original HiC-maps/ graph nodes.

    :param parameters: dictionary with parameters set in parameters.json file
    :param isdict: creates dictionary with TAD regions if isdict = True, otherwise np.array
    :return solution: TADs in original HiC-maps/ graph nodes called by Arrowhead method (Ground truth) separated for chromosomes and cell line
    :return solution_nodes: TAD classification for genomic bins in original HiC-maps/ graph nodes called by Arrowhead method (Ground truth) separated for chromosomes and cell line
    '''
    solution = []
    solution_nodes = []

    for cell_line in parameters["cell_lines"]:
        solution_cell_line = []
        solution_nodes_cell_line = []
        if isdict:
            solution_dict = dict.fromkeys(["Arrowhead"])
            solution_dict["Arrowhead"] = dict.fromkeys(parameters["chromosomes_str_short"])

        for chromosome in parameters["chromosomes_str_short"]:

            if cell_line == "GM12878":
                df_solution = pd.read_csv(parameters["arrowhead_solution_GM12878"], delimiter="\t")
            elif cell_line == "IMR-90":
                df_solution = pd.read_csv(parameters["arrowhead_solution_IMR-90"], delimiter="\t")
            df_solution = df_solution[(df_solution["chr1"] == chromosome) & (df_solution["chr2"] == chromosome)]

            df_solution["x1"] = df_solution["x1"].apply(lambda x: np.int(round(x/parameters["scaling_factor"], 0)))
            df_solution["x2"] = df_solution["x2"].apply(lambda x: np.int(round(x/parameters["scaling_factor"], 0)))

            solution_chromosome = []

            for index, row in df_solution.iterrows():
                #df_solution_tuples.append((row["x1"], row["x2"]))
                #df_solution_tuples.append((row["x2"], row["x1"]))
                solution_chromosome.append(list(range(row["x1"], row["x2"]+1)))

            if isdict:
                solution_chromosome.sort()
                solution_dict["Arrowhead"][chromosome] = solution_chromosome
            else:
                solution_nodes_chromosome = np.unique([node for tad_list in solution_chromosome for node in tad_list])

                solution_cell_line.append(solution_chromosome)
                solution_nodes_cell_line.append(solution_nodes_chromosome)

        if isdict:
            solution.append(solution_dict)
        else:
            solution.append(solution_cell_line)
            solution_nodes.append(solution_nodes_cell_line)

    return solution, solution_nodes


def arrowhead_predicted_tad_list(nodes, df_solution_nodes_list):
    '''

    :param nodes:
    :param df_solution_nodes_list:
    :return:
    '''

    predicted_tad = [1 if node in df_solution_nodes_list else 0 for node in nodes]
    predicted_tad = np.array(predicted_tad)

    return predicted_tad


def one_hot_encode_labels_solution(parameters, adjacency_matrices_list, solution_nodes):
    '''
    Function creates a one-hot encoded list of TAD (1) or No-TAD (0) based on calling of Arrowhead for all nodes in a adjacency matrix in adjacency_matrices_list.

    :param adjacency_matrices_list: adjacency matrices separated for chromosomes and cell line
    :param solution_nodes: TAD classification for genomic bins in original HiC-maps/ graph nodes by Arrowhead method (Ground truth) separated for chromosomes and cell line
    :return:
    '''

    solution_nodes_one_hot = []

    if parameters["labels_borders_region"] == "region":
        for cell_line, solution_nodes_cell_line in zip(adjacency_matrices_list, solution_nodes):
            solution_nodes_one_hot_cell_line = []
            for chromosome, solution_nodes_cell_line_chromosome in zip(cell_line, solution_nodes_cell_line):
                solution_nodes_one_hot_cell_line_chromosome = np.zeros([chromosome.shape[0],])
                solution_nodes_one_hot_cell_line_chromosome[solution_nodes_cell_line_chromosome] = 1
                solution_nodes_one_hot_cell_line.append(solution_nodes_one_hot_cell_line_chromosome)

            solution_nodes_one_hot.append(solution_nodes_one_hot_cell_line)
    elif parameters["labels_borders_region"] == "borders":
        for cell_line, solution_nodes_cell_line in zip(adjacency_matrices_list, solution_nodes):
            solution_nodes_one_hot_cell_line = []
            for chromosome, solution_nodes_cell_line_chromosome in zip(cell_line, solution_nodes_cell_line):
                solution_nodes_one_hot_cell_line_chromosome = np.zeros([chromosome.shape[0], ])
                for index_solution_node, solution_node in enumerate(solution_nodes_cell_line_chromosome):
                    if solution_node-1 != solution_nodes_cell_line_chromosome[index_solution_node-1]:
                        solution_nodes_one_hot_cell_line_chromosome[solution_node] = 1
                    if solution_nodes_cell_line_chromosome[index_solution_node]+1 not in solution_nodes_cell_line_chromosome:
                        solution_nodes_one_hot_cell_line_chromosome[solution_node] = 1
                solution_nodes_one_hot_cell_line.append(solution_nodes_one_hot_cell_line_chromosome)
            solution_nodes_one_hot.append(solution_nodes_one_hot_cell_line)
    else:
        raise NotImplementedError("Please provide 'region' or 'borders' as the ")

    return solution_nodes_one_hot


def save_arrowhead_dict(list_solution_dict, baseoutput_path, cell_lines, scaling_factor):
    """
    saves a tad dict with the arrowhead solutions
    :param list_solution_dict: a list of dicts with Arrowhead tad calling
    :param baseoutput_path: output path to which the folders /evaluation/tad_dicts/ are added
    :param cell_lines: list of cell lines that correspond to the ones from the dicts
    :param scaling_factor: resolution either 100000 or 25000
    :return:
    """
    for count, cell_line in enumerate(cell_lines):
        output_dir = os.path.join(baseoutput_path, "evaluation/tad_dicts/")
        output_path = output_dir + "Arrowhead_" + cell_line + "_" + str(int(scaling_factor/1000)) + "kb_dict.pickle"
        cell_line_dict = list_solution_dict[count]
        with open(output_path, 'wb') as handle:
            pickle.dump(cell_line_dict, handle, protocol=pickle.DEFAULT_PROTOCOL)
    return


def load_french_team_labels(parameters, adjacency_matrices_list, adjacency_matrices_source_information_list):
    '''
    Function loads dataframes with TAD classification used by french team (Called by various methods.) (Ground truth) and extracts the TADs in original HiC-maps/ graph nodes and the TAD classification for genomic bins in original HiC-maps/ graph nodes.

    :param parameters: dictionary with parameters set in parameters.json file
    :return solution: TADs in original HiC-maps/ graph nodes used by french team (Ground truth) separated for chromosomes and cell line
    :return solution_nodes: TAD classification for genomic bins in original HiC-maps/ graph nodes used by french team (Ground truth) separated for chromosomes and cell line
    '''

    if len(adjacency_matrices_source_information_list) != 1 and adjacency_matrices_source_information_list[0][0].split("-")[0] != "GM12878":
        raise NotImplementedError("Labels by french team are currently only available for GM12878 dataset.")

    labels_french_team_df = pd.read_csv(parameters["path_labels_french_team"], delimiter=" ", index_col=[0])

    labels_french_team = []

    for adjacency_matrices_list_cell_line, adjacency_matrices_source_information_list_cell_line in zip(adjacency_matrices_list, adjacency_matrices_source_information_list):
        labels_french_team.append([])
        for adjacency_matrix, adjacency_matrix_source_information in zip(adjacency_matrices_list_cell_line, adjacency_matrices_source_information_list_cell_line):
            labels_french_team_df_chr_subset = labels_french_team_df[labels_french_team_df["chr"] == "chr" + adjacency_matrix_source_information.split("-")[-1]]
            labels_french_team_chr_subset = []
            for index, row in labels_french_team_df_chr_subset.iterrows():
                labels_french_team_chr_subset += list(range(row["start"], row["end"] + 1))

            print(f"Chromosome {adjacency_matrix_source_information.split('-')[-1]}: {len(labels_french_team_chr_subset) - len(set(labels_french_team_chr_subset))} marked as duplicates.")
            labels_french_team_chr_subset = list(np.unique(labels_french_team_chr_subset)[np.where(np.unique(labels_french_team_chr_subset) < adjacency_matrix.shape[0])[0]])
            labels_french_team[0].append(labels_french_team_chr_subset)

    return labels_french_team


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create one-hot encoded list (1 for TAD, 0 for non-TAD) array and dicts of true labels called by Arrowhead for genomic bins for chosen chromosomes and cell lines.')
    parser.add_argument("--path_parameters_json", help="path to JSON with parameters.", type=str, required=True)
    args = parser.parse_args()
    path_parameters_json = args.path_parameters_json

    parameters = load_parameters(path_parameters_json)
    os.makedirs(parameters["output_directory"], exist_ok=True)
    os.makedirs(os.path.join(parameters["output_directory"], "preprocessing"), exist_ok=True)
    os.makedirs(os.path.join(parameters["output_directory"], "evaluation/tad_dicts/"), exist_ok=True)

    logger = set_up_logger('arrowhead_solution', parameters)
    logger.debug('Start arrowhead_solution logger.')

    parameters["chromosomes_int"], parameters["chromsomes_str_long"], parameters["chromosomes_str_short"] = generate_chromosome_lists(parameters)

    list_solution_dict, _ = load_arrowhead_solution(parameters, isdict=True)
    save_arrowhead_dict(list_solution_dict, parameters["output_directory"], parameters["cell_lines"], parameters["scaling_factor"])

    _, df_solution_nodes_list = load_arrowhead_solution(parameters)
    data = load_data(parameters)
    predicted_tad = arrowhead_predicted_tad_list(list(range(0, data.edge_index.shape[0])), df_solution_nodes_list)
    save_tad_list(parameters, predicted_tad, "Arrowhead")

