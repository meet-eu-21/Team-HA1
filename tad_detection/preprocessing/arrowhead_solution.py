import sys
sys.path.insert(1, './tad_detection/')
sys.path.insert(1, './tad_detection/preprocessing/')
sys.path.insert(1, './tad_detection/model/')
sys.path.insert(1, './tad_detection/evaluation/')

import pandas as pd
import os
import numpy as np
import argparse
from utils_general import load_parameters, set_up_logger
from model.utils_model import load_data, save_tad_list
from evaluation.utils_evaluate import generate_chromosome_lists
import pickle


def load_arrowhead_solution(parameters):
    '''
    Function loads dataframes with TAD classification called by Arrowhead method (Ground truth) and extracts the TADs in original HiC-maps/ graph nodes and the TAD classification for genomic bins in original HiC-maps/ graph nodes.

    :param parameters: dictionary with parameters set in parameters.json file
    :return solution: TADs in original HiC-maps/ graph nodes called by Arrowhead method (Ground truth) separated for chromosomes and cell line
    :return solution_nodes: TAD classification for genomic bins in original HiC-maps/ graph nodes called by Arrowhead method (Ground truth) separated for chromosomes and cell line
    '''

    solution = []
    solution_nodes = []

    for cell_line in parameters["cell_lines"]:
        solution_cell_line = []
        solution_nodes_cell_line = []
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

            solution_nodes_chromosome = np.unique([node for tad_list in solution_chromosome for node in tad_list])

            solution_cell_line.append(solution_chromosome)
            solution_nodes_cell_line.append(solution_nodes_chromosome)

        solution.append(solution_cell_line)
        solution_nodes.append(solution_nodes_cell_line)

        '''
        arrowhead_solution = []

        for chrom in chromosome_list:
            df_solution = pd.read_csv(f"{data_prefix}meeteu/www.lcqb.upmc.fr/meetu/dataforstudent/TAD/GSE63525_GM12878_primary+replicate_Arrowhead_domainlist.txt", delimiter="\t")
        
            df_solution = df_solution[(df_solution["chr1"] == str(chrom)) & (df_solution["chr2"] == str(chrom))]
            df_solution["x1"] = df_solution["x1"].apply(lambda x: np.int(round(x/100000, 0)))
            df_solution["x2"] = df_solution["x2"].apply(lambda x: np.int(round(x/100000, 0)))
            
            df_solution_ = []
            
            for index, row in df_solution.iterrows():
                df_solution_.append(np.arange(row["x1"], row["x2"]+1).tolist())
            
            s_solution = sorted(np.asarray(df_solution_).squeeze())
            s_solution = [x for x in s_solution if x != []]
            
            arrowhead_solution.append(s_solution)
        '''

    return solution, solution_nodes


def load_arrowhead_solution_dict(parameters):
    list_dicts_cell_lines = []
    for cell_line in parameters["cell_lines"]:

        solution_dict = dict.fromkeys(["Arrowhead"])
        solution_dict["Arrowhead"] = dict.fromkeys(parameters["chromosomes_str_short"])

        for chromosome in parameters["chromosomes_str_short"]:
            if cell_line == "GM12878":
                df_solution = pd.read_csv(parameters["arrowhead_solution_GM12878"], delimiter="\t")
            elif cell_line == "IMR-90":
                df_solution = pd.read_csv(parameters["arrowhead_solution_IMR-90"], delimiter="\t")
            else:
                df_solution = pd.read_csv(parameters["arrowhead_solution_" + cell_line], delimiter="\t")

            df_solution = df_solution[(df_solution["chr1"] == chromosome) & (df_solution["chr2"] == chromosome)]

            df_solution["x1"] = df_solution["x1"].apply(lambda x: np.int(round(x/parameters["scaling_factor"], 0)))
            df_solution["x2"] = df_solution["x2"].apply(lambda x: np.int(round(x/parameters["scaling_factor"], 0)))

            df_sol = df_solution.iloc[:, 0:3]
            df_sol = df_sol.sort_values(by=['x1'])

            solution_chromosome = []

            for index, row in df_sol.iterrows():
                list_tads = list(range(row["x1"], row["x2"]+1))
                solution_chromosome.append(list_tads)
                # print(index)
            # print(solution_chromosome[0:10])
            solution_chromosome.sort()
            # print(solution_chromosome)
            solution_dict["Arrowhead"][chromosome] = solution_chromosome
        list_dicts_cell_lines.append(solution_dict)

    return list_dicts_cell_lines


def arrowhead_predicted_tad_list(nodes, df_solution_nodes_list):
    '''

    :param nodes:
    :param df_solution_nodes_list:
    :return:
    '''

    predicted_tad = [1 if node in df_solution_nodes_list else 0 for node in nodes]
    predicted_tad = np.array(predicted_tad)

    return predicted_tad


def one_hot_encode_arrowhead_solution(adjacency_matrices_list, solution_nodes):
    '''
    Function creates a one-hot encoded list of TAD (1) or No-TAD (0) based on calling of Arrowhead for all nodes in a adjacency matrix in adjacency_matrices_list.

    :param adjacency_matrices_list: adjacency matrices separated for chromosomes and cell line
    :param solution_nodes: TAD classification for genomic bins in original HiC-maps/ graph nodes by Arrowhead method (Ground truth) separated for chromosomes and cell line
    :return:
    '''

    solution_nodes_one_hot = []

    for cell_line, solution_nodes_cell_line in zip(adjacency_matrices_list, solution_nodes):
        solution_nodes_one_hot_cell_line = []
        for chromosome, solution_nodes_cell_line_chromosome in zip(cell_line, solution_nodes_cell_line):
            solution_nodes_one_hot_cell_line_chromosome = np.zeros([chromosome.shape[0],])
            solution_nodes_one_hot_cell_line_chromosome[solution_nodes_cell_line_chromosome] = 1
            solution_nodes_one_hot_cell_line.append(solution_nodes_one_hot_cell_line_chromosome)

        solution_nodes_one_hot.append(solution_nodes_one_hot_cell_line)

    return solution_nodes_one_hot


def save_dict(list_solution_dict, baseoutput_path, cell_lines, scaling_factor):
    for count, cell_line in enumerate(cell_lines):
        output_dir = os.path.join(baseoutput_path, "evaluation/results/")
        output_path = output_dir + "Arrowhead_" + cell_line + "_" + str(int(scaling_factor/1000)) + "kb_dict.p"
        cell_line_dict = list_solution_dict[count]
        with open(output_path, 'wb') as handle:
            pickle.dump(cell_line_dict, handle, protocol=pickle.DEFAULT_PROTOCOL)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_parameters_json", help=" to JSON with parameters.", type=str, required=True)
    args = parser.parse_args()
    path_parameters_json = args.path_parameters_json

    parameters = load_parameters(path_parameters_json)
    os.makedirs(parameters["output_directory"], exist_ok=True)
    os.makedirs(os.path.join(parameters["output_directory"], "preprocessing"), exist_ok=True)
    os.makedirs(os.path.join(parameters["output_directory"], "evaluation/results/"), exist_ok=True)

    logger = set_up_logger('arrowhead_solution', parameters)
    logger.debug('Start arrowhead_solution logger.')

    parameters["chromosomes_int"], parameters["chromsomes_str_long"], parameters["chromosomes_str_short"] = generate_chromosome_lists(parameters)

    list_solution_dict = load_arrowhead_solution_dict(parameters)
    save_dict(list_solution_dict, parameters["output_directory"], parameters["cell_lines"], parameters["scaling_factor"])

    _, df_solution_nodes_list = load_arrowhead_solution(parameters)
    data = load_data(parameters)
    predicted_tad = arrowhead_predicted_tad_list(list(range(0, data.edge_index.shape[0])), df_solution_nodes_list)
    save_tad_list(parameters, predicted_tad, "Arrowhead")

