import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles
from matplotlib_venn import venn3, venn3_circles

file_TopDomSol = "/Users/Charlotte/MeetEU/TopResults/100kb/GM12878/ 100kb_chr7_adj.csv.bed"

file_AH = "/Users/Charlotte/TAD_Data/GSE63525_GM12878_primary+replicate_Arrowhead_domainlist.txt"
folder_TopDom = "/Users/Charlotte/MeetEU/TopResults/100kb/GM12878/"
chromosomes_str_short = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16",
                          "17", "18", "19", "20", "21", "22"]   # without X
scaling_factor = 100000

############################################
for chromosome in chromosomes_str_short:
    ## arrowhead_solution
    df_solution = pd.read_csv(os.path.join(file_AH), delimiter="\t")

    df_solution_chr = df_solution[(df_solution["chr1"] == chromosome) & (df_solution["chr2"] == chromosome)]

    df_solution_chr["x1"] = df_solution_chr["x1"].apply(lambda x: np.int(round(x/scaling_factor, 0)))
    df_solution_chr["x2"] = df_solution_chr["x2"].apply(lambda x: np.int(round(x/scaling_factor, 0)))
    #
    df_sol = df_solution_chr.iloc[:, 0:3]
    df_sol = df_sol.sort_values(by=['x1'])

    print(df_sol)

    # arrowhead_solution_lists = [[620, 621, 522, ..., 627], [1526, ..., 1543], [6], [6, 7, 8], ..]
    arrowhead_solution_list = []
    for row in range(0,len(df_sol)):
        r = range(df_sol.iloc[row, 1], df_sol.iloc[row,2])
        l = [*r]
        arrowhead_solution_list.append(l)
    print(arrowhead_solution_list[0:10])
    flat_arrow = set([arr_td_item for arr_td_sub in arrowhead_solution_list for arr_td_item in arr_td_sub])

    ################################################
    # topdom solution
    file_td = folder_TopDom + " 100kb_chr" + chromosome + "_adj.csv.bed"
    df_topdom = pd.read_csv(file_td, delimiter="\t", header=None)

    df_topdom.iloc[:,1] = df_topdom.iloc[:,1].apply(lambda x: np.int(round(x/scaling_factor, 0)))
    df_topdom.iloc[:,2] = df_topdom.iloc[:,2].apply(lambda x: np.int(round(x/scaling_factor, 0)))

    # df_topdom.columns
    df_topdom = df_topdom.rename(columns={0:"chr", 1:"x1", 2:"x2",3:"label"})

    df_topdom = df_topdom.astype({"x1":'int', "x2":'int'})
    print(df_topdom)
    df_td = df_topdom[df_topdom["label"] != "boundary"]
    print(df_td)
    td_solution_list = []
    for row in range(0,len(df_td)):
        r = range(df_td.iloc[row, 1], df_td.iloc[row,2])
        l = [*r]
        td_solution_list.append(l)
    print(td_solution_list[0:10])
    flat_td = set([top_td_item for top_td_sub in td_solution_list for top_td_item in top_td_sub])

    ############################################################################################
    ## jaccard index
    predicted_tads_per_tad_prediction_methods = []

    predicted_tads_per_tad_prediction_methods.append(flat_arrow)
    predicted_tads_per_tad_prediction_methods.append(flat_td)

    tad_prediction_methods = ["arrowhead", "topdom"]

    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
    venn2([flat_arrow, flat_td], set_labels=("Arrowhead", "TopDom"), alpha=0.75, set_colors=('c', 'tab:orange'))
    venn2_circles([flat_arrow, flat_td], lw=0.5)
    plt.title("chromosome " + chromosome)
    # plt.show()
    plt.savefig("/Users/Charlotte/MeetEU/tad_detection/evaluation/results/AH_TopDom/Venn_Chr" + chromosome +"_TopDom_ArrowHead.png")
    plt.clf()
