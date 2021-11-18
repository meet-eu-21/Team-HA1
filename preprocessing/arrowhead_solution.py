import pandas as pd
import os
import numpy as np

def load_arrowhead_solution(parameters):

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