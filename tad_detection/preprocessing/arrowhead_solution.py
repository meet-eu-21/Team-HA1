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
from utils_model import load_data, save_tad_list

def load_arrowhead_solution(parameters):

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

    return solution, solution_nodes

def arrowhead_predicted_tad_list(nodes, df_solution_nodes_list):

    predicted_tad = [1 if node in df_solution_nodes_list else 0 for node in nodes]
    predicted_tad = np.array(predicted_tad)

    return predicted_tad

def one_hot_encode_arrowhead_solution(adjacency_matrices_list, solution_nodes):

    solution_nodes_one_hot = []

    for cell_line, solution_nodes_cell_line in zip(adjacency_matrices_list, solution_nodes):
        solution_nodes_one_hot_cell_line = []
        for chromosome, solution_nodes_cell_line_chromosome in zip(cell_line, solution_nodes_cell_line):
            solution_nodes_one_hot_cell_line_chromosome = np.zeros([chromosome.shape[0],])
            solution_nodes_one_hot_cell_line_chromosome[solution_nodes_cell_line_chromosome] = 1
            solution_nodes_one_hot_cell_line.append(solution_nodes_one_hot_cell_line_chromosome)

        solution_nodes_one_hot.append(solution_nodes_one_hot_cell_line)

    return solution_nodes_one_hot

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_parameters_json", help=" to JSON with parameters.", type=str, required=True)
    args = parser.parse_args()
    path_parameters_json = args.path_parameters_json

    parameters = load_parameters(path_parameters_json)
    os.makedirs(parameters["output_directory"], exist_ok=True)
    os.makedirs(os.path.join(parameters["output_directory"], "preprocessing"), exist_ok=True)
    logger = set_up_logger('arrowhead_solution', parameters)
    logger.debug('Start arrowhead_solution logger.')

    data = load_data(parameters, 'cpu')

    _, df_solution_nodes_list = load_arrowhead_solution(parameters)
    predicted_tad = arrowhead_predicted_tad_list(list(range(0, data.edge_index.shape[0])), df_solution_nodes_list)

    save_tad_list(parameters, predicted_tad, "Arrowhead")