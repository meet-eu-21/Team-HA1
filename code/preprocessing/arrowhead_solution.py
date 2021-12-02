import sys
sys.path.insert(1, './preprocessing/')
sys.path.insert(1, './model/')
sys.path.insert(1, './evaluation/')

import pandas as pd
import os
import numpy as np
import argparse
from utils_preprocessing import load_parameters
from utils_model import load_data, save_tad_list

def load_arrowhead_solution(parameters):

    #TODO
    #AUF MEHRERE ZELLEN &CHROMOSOMEN ANPASSEN

    df_solution_tuples_list = []
    df_solution_nodes_list = []

    for cell_line in parameters["cell_lines"]:
        for chromosome in parameters["chromosomes_str_short"]:

            df_solution = pd.read_csv(os.path.join(parameters["arrowhead_solution_directory"], "GSE63525_" + cell_line + "_primary+replicate_Arrowhead_domainlist.txt"), delimiter="\t")
            df_solution = df_solution[(df_solution["chr1"] == chromosome) & (df_solution["chr2"] == chromosome)]

            df_solution["x1"] = df_solution["x1"].apply(lambda x: np.int(round(x/parameters["scaling_factor"], 0)))
            df_solution["x2"] = df_solution["x2"].apply(lambda x: np.int(round(x/parameters["scaling_factor"], 0)))

            df_solution_tuples = []

            for index, row in df_solution.iterrows():
                df_solution_tuples.append((row["x1"], row["x2"]))
                df_solution_tuples.append((row["x2"], row["x1"]))

            df_solution_tuples_list.append(df_solution_tuples)

            df_solution_nodes = []

            for index, row in df_solution.iterrows():
                df_solution_nodes.append(row["x1"])
                df_solution_nodes.append(row["x2"])

            df_solution_nodes_list.append(df_solution_nodes)

    return df_solution_tuples_list, df_solution_nodes_list

def arrowhead_predicted_tad_list(nodes, df_solution_nodes_list):

    predicted_tad = [1 if node in df_solution_nodes_list else 0 for node in nodes]
    predicted_tad = np.array(predicted_tad)

    return predicted_tad

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_parameters_json", help=" to JSON with parameters.", type=str, required=True)
    args = parser.parse_args()
    path_parameters_json = args.path_parameters_json

    parameters = load_parameters(path_parameters_json)
    os.makedirs(parameters["output_directory"], exist_ok=True)
    os.makedirs(os.path.join(parameters["output_directory"], "preprocessing"), exist_ok=True)

    data = load_data(parameters, 'cpu')

    _, df_solution_nodes_list = load_arrowhead_solution(parameters)
    predicted_tad = arrowhead_predicted_tad_list(list(range(0, data.edge_index.shape[0])), df_solution_nodes_list)

    save_tad_list(parameters, predicted_tad, "Arrowhead")