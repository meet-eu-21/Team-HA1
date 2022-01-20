import pandas as pd
import numpy as np
import os
from preprocessing.arrowhead_solution import load_arrowhead_solution
import argparse
from utils_general import load_parameters

def translate_bed_to_npy(folder):
    # files = folder
    cell_lines = ["GM12878"]
    solution_cell_line=np.array([])
    for cell_line in cell_lines:
        bedfiles = []
        for file in os.listdir(folder):
            if file.endswith(".bed"):
                print(file)
                bedfiles.append(file)

        solution_np = np.array([])
        # solution_np = []
        for bedfile in bedfiles:
            fullfile = folder + bedfile
            df_topdom = pd.read_csv(fullfile, delimiter="\t", header=None)
            resolution = 100000     #### # add to parameters?
            df_topdom.iloc[:,1] = df_topdom.iloc[:,1].apply(lambda x: np.int(round(x/resolution, 0)))
            df_topdom.iloc[:,2] = df_topdom.iloc[:,2].apply(lambda x: np.int(round(x/resolution, 0)))
            df_topdom = df_topdom.rename(columns={0:"chr", 1:"x1", 2:"x2",3:"label"})
            df_topdom = df_topdom.astype({"x1":'int', "x2":'int'})

            df_td = df_topdom[df_topdom["label"] != "boundary"]
            td_solution_list = []
            for row in range(0,len(df_td)):
                r = range(df_td.iloc[row, 1], df_td.iloc[row,2])
                l = [*r]
                td_solution_list.append(l)

            split = bedfile.split("_")[1]
            print(split)
            array_sol = np.array([split, td_solution_list], dtype=object)
            print(array_sol)
            solution_np = np.append(solution_np, array_sol, axis=0)


        single_cell_sol = np.array([cell_line, solution_np], dtype=object)
        solution_cell_line = np.append(solution_cell_line, single_cell_sol)


    np.save("/Users/Charlotte/MeetEU/tad_detection/evaluation/results/TopDomGM12878_100kb_v2.npy", solution_cell_line)
    print("save")
    return solution_cell_line


def label_solution(ah_solution, parameters):
    chrom_str = parameters["chromosomes_str_long"]
    cell_line = parameters["cell_line"]
    # ah_solution_label = np.array([cell_line])
    all_chrom = []
    for count, chr in enumerate(chrom_str):
        chrom_tads = np.array([chr, ah_solution[0][count]])
        all_chrom = np.append(all_chrom, chrom_tads)
    ah_solution_label = np.array([cell_line, all_chrom])

    np.save("/Users/Charlotte/MeetEU/tad_detection/evaluation/results/AH_GM12878_100kb.npy", ah_solution_label)
    return ah_solution_label


def arrowhead_to_npy(parameters):
    ## arrowhead_solution
    file_arrowhead = parameters["arrowhead_path"]
    df_solution = pd.read_csv(os.path.join(file_arrowhead), delimiter="\t")
    cell_line = parameters["cell_line"]
    chr_short = parameters["chromosomes_str_short"]
    all_chrom = []
    for chromosome in chr_short:
        df_solution_chr = df_solution[(df_solution["chr1"] == chromosome) & (df_solution["chr2"] == chromosome)]

        scaling_factor = parameters["scaling_factor"]
        df_solution_chr["x1"] = df_solution_chr["x1"].apply(lambda x: np.int(round(x / scaling_factor, 0)))
        df_solution_chr["x2"] = df_solution_chr["x2"].apply(lambda x: np.int(round(x / scaling_factor, 0)))

        # df_solution_tuples = []
        #
        # for index, row in df_solution_chr.iterrows():
        #     df_solution_tuples.append((row["x1"], row["x2"]))
        # df_solution_nodes = []
        #
        # for index, row in df_solution_chr.iterrows():
        #     df_solution_nodes.append(row["x1"])
        #     df_solution_nodes.append(row["x2"])

        #
        df_sol = df_solution_chr.iloc[:, 0:3]
        df_sol = df_sol.sort_values(by=['x1'])

        arrowhead_solution_list = []
        for row in range(0, len(df_sol)):
            r = range(df_sol.iloc[row, 1], df_sol.iloc[row, 2])
            l = [*r]
            arrowhead_solution_list.append(l)
        arrowhead_solution_list = np.array(["chr" + chromosome, arrowhead_solution_list])
        all_chrom = np.append(all_chrom, arrowhead_solution_list)

    ah_solution_label = np.array([cell_line, all_chrom])
    np.save("/Users/Charlotte/MeetEU/tad_detection/evaluation/results/AH_GM12878_100kb_v2.npy", ah_solution_label)


    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_parameters_json", help=" to JSOn with parameters.", type=str, required=True)
    args = parser.parse_args()
    path_parameters_json = args.path_parameters_json
    parameters = load_parameters(path_parameters_json)


    # cell_line_array = test_npy()
    # cell_line_array
    folder = "/Users/Charlotte/MeetEU/TopResults/100kb/GM12878/"
#     # /Users/Charlotte/VSC/MeetEU/TopResults/100kb/GM12878/ 100kb_chr1_adj.csv.bed
    solution_cell_line = translate_bed_to_npy(folder)
    print(solution_cell_line)

    ah_solution_label = arrowhead_to_npy(parameters)


    # ah_solution, ah_solution_flat = load_arrowhead_solution(parameters)
    # ah_solution_label = label_solution(ah_solution, parameters)



    # def test_npy():
    #     cell_lines = ["A", "B"]
    #     chromosomes = ["1", "2", "3"]
    #     final_array = np.array(["results"], dtype=object)
    #     print(final_array)
    #     for cell_line in cell_lines:
    #         cell_line_array = np.array([cell_line], dtype=object) #np.array(["A"])
    #         print(cell_line_array)
    #         for chrom in chromosomes:
    #             tad_array = [[1,2,3], [5,6,7]]
    #             chromosome_array = np.array([chrom, tad_array], dtype=object) #np.array(["1", [[123...]] ])
    #             print(chromosome_array)
    #
    #             cell_line_array = np.append(cell_line_array, chromosome_array)
    #         final_array = np.append(final_array, cell_line_array)

    # final_array = np.array(["results", cell_line_array])
    # cell_line_array = np.array["cell line", chromosome_array]
    # chromosome_array = np.array["chr", tad_list]
    # return final_array