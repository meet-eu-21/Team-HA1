import sys
sys.path.insert(1, './tad_detection/')
sys.path.insert(1, './tad_detection/preprocessing/')
sys.path.insert(1, './tad_detection/model/')
sys.path.insert(1, './tad_detection/evaluation/')

import numpy as np 
import pandas as pd
from utils_preprocessing import generate_chromosome_lists
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_parameters_json", help=" to JSON with parameters.", type=str, required=True)
    args = parser.parse_args()
    path_parameters_json = args.path_parameters_json

    parameters = load_parameters(path_parameters_json)

    parameters["chromosomes_int"], parameters["chromosomes_str_long"], parameters["chromosomes_str_short"] = generate_chromosome_lists(parameters)

    adjacency_matrices_list, adjacency_matrices_source_information_list

    # only for chr7 at the moment
    file = "/home/stefanie/data/HiC/GM12878/100kb_resolution_intrachromosomal/chr7_100kb.RAWobserved"
    print("file", file)




    #from preprocessing
    lines = []
    with open(file,'r') as f:
        lines_sub = f.read().splitlines()
    print(len(lines_sub))
    lines = lines + lines_sub
    f.close()

    data = [i.split('\t') for i in lines]
    col0 = [row[0] for row in data]
    un0 = np.unique(col0)

    col1 = [row[1] for row in data]
    un1 = np.unique(col1)

    z = list(zip(*data))
    print(len(z))

    # parameters["scaling_factor"] = 100000
    row_indices = np.array(list(map(int, z[0])))
    row_indices = row_indices / 100000
    row_indices = row_indices.astype(int)

    # parameters["scaling_factor"] = 100000
    column_indices = np.array(list(map(int, z[1])))
    column_indices = column_indices / 100000
    column_indices = column_indices.astype(int)

    values = list(map(float, z[2]))

    m = max(row_indices) + 1
    n = max(column_indices) + 1
    p = max([m, n])

    print("max row indices +1 ", m)
    print("max column indices +1 ", n)
    print("max m, n ", p)

    adjacency_matrix = np.zeros((p, p))

    adjacency_matrix[row_indices, column_indices] = values
    print("type matrix ", type(adjacency_matrix))

    np.savetxt("chr7adj.csv", adjacency_matrix, delimiter="\t")
    print('saved')

    ############
    # add columns
    seven = np.full((p, 1), 7, dtype=int)
    print(seven)
    print("length seven ", len(seven), seven.shape)

    binstart = np.arange(0,p*100000,100000)
    binstart =binstart.T
    binstart = binstart.reshape(p,1)
    print(binstart.shape)

    binend = np.arange(100000,p*100000+100000,100000)
    binend = binend.T
    binend=binend.reshape(p,1)
    print(binend.shape)

    print("binend ", binend.shape, binend)
    print("binstart ", binstart.shape, binstart)
    adj_matr3 = np.hstack((seven, binstart, binend, adjacency_matrix))
    np.savetxt("chr7adj3.csv", adj_matr3, delimiter="\t")
    print('saved')