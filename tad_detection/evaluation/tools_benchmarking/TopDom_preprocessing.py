import numpy as np
import os
from tad_detection.evaluation.utils_evaluate import generate_chromosome_lists
import argparse
from tad_detection.utils_general import load_parameters

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocessing of Hi-C data for TopDom run.')
    parser.add_argument("--path_parameters_json", help="path to JSON with parameters.", type=str, required=True)
    args = parser.parse_args()
    path_parameters_json = args.path_parameters_json

    parameters = load_parameters(path_parameters_json)
    _,_, chromosome_names = generate_chromosome_lists(parameters)

    #full output EXAMPLE: "home/charlotte/adj_matrix_cell_line/resolution/resolution_chrnum_adj.csv"
    # /cell_line/resolution/chromosome

    for cell_line in parameters["cell_lines"]:
        for resolution in parameters["resolutions"]:
            for chromosome_name in chromosome_names:
                #path = inputpath + cell_line + "/" + resolution + "kb_resolution_intrachromosomal/"
                lines = []
                #filepath = path + chromosome_name + "_" + resolution + "kb.RAWobserved"
                inputpath = parameters["inputpath"]
                if cell_line == "GM12878":
                    filepath = inputpath + cell_line + "/" + resolution + "kb_resolution_intrachromosomal/" + chromosome_name + "_" + resolution + "kb.RAWobserved"
                elif cell_line == "HUVEC" and resolution == "100":
                    filepath = inputpath + cell_line + "/" + resolution + "kb_resolution_intrachromosomal/" + resolution + "kb_resolution_intrachromosomal/" + chromosome_name + "/MAPQGE30/" + chromosome_name + "_" + resolution + "kb.RAWobserved"
                else:
                    filepath = inputpath + cell_line + "/" + resolution + "kb_resolution_intrachromosomal/" + chromosome_name + "/MAPQGE30/" + chromosome_name + "_" + resolution + "kb.RAWobserved"

                with open(filepath,'r') as f:
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

                res = int(resolution) * 1000

                row_indices = np.array(list(map(int, z[0])))
                row_indices = row_indices / res
                row_indices = row_indices.astype(int)

                column_indices = np.array(list(map(int, z[1])))
                column_indices = column_indices / res
                column_indices = column_indices.astype(int)

                values = list(map(float, z[2]))

                m = max(row_indices) + 1
                n = max(column_indices) + 1
                p = max([m, n])

                # print("max row indices +1 ", m)
                # print("max column indices +1 ", n)
                # print("max m, n ", p)

                adjacency_matrix = np.zeros((p, p))
                adjacency_matrix[row_indices, column_indices] = values

                ## add three initial columns
                num = chromosome_name[3:]
                if num != 'X':
                    num = int(num)
                else:
                    num = int(23)

                nums = np.full((p, 1), num, dtype=int)
                print(nums)
                print("length nums ", len(nums), nums.shape)

                binstart = np.arange(0,p*res,res)
                binstart = binstart.T
                binstart = binstart.reshape(p,1)

                binend = np.arange(res,p*res+res,res)
                binend = binend.T
                binend = binend.reshape(p,1)

                print(len(nums), len(binstart), len(binend), len(adjacency_matrix))
                adj_matr3 = np.hstack((nums, binstart, binend, adjacency_matrix))


                folder = parameters["outputpath"] + "adj_matrix_" + cell_line + "/" + resolution + "kb/"      #"home/charlotte/adj_matrix_cell_line/25kb/"
                os.makedirs(folder, exist_ok=True)

                savepath = folder + resolution + "kb_" + chromosome_name + "_adj.csv"   #"home/charlotte/adj_matrix_cell_line/25kb/25kb_chrnum_adj.csv"
                np.savetxt(savepath, adj_matr3, delimiter="\t")
                print("save: ", savepath)


