import sys
sys.path.insert(1, './tad_detection/')
sys.path.insert(1, './tad_detection/preprocessing/')
sys.path.insert(1, './tad_detection/model/')
sys.path.insert(1, './tad_detection/evaluation/')

import logging
logger = logging.getLogger('evaluation')

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from venn import venn
from matplotlib_venn import venn2, venn2_circles
from matplotlib_venn import venn3, venn3_circles
import json
import logging
from itertools import combinations, combinations_with_replacement
import pickle
from gcMapExplorer import lib as gmlib


def generate_chromosome_lists(parameters):
    '''
    Function generates lists of chromosomes in different representations (integers and strings) based on the input in the parameters.json file for the usage in other functions.

    :param parameters: dictionary with parameters set in parameters.json file
    :return chromosomes_int: integer list of chromosomes
    :return chromosomes_str_long: string list of chromosomes in the format "chr<chromosome>"
    :return chromosomes_str_short: string list of chromosomes in the format "<chromosome>"
    '''

    if parameters["chromosomes"] == "all":
        chromsomes_int = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, "X"]
        chromosomes_str_long = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10",
                                "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", "chr18", "chr19",
                                "chr20", "chr21", "chr22", "chrX"]
        chromosomes_str_short = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16",
                                 "17", "18", "19", "20", "21", "22", "X"]
    else:
        chromsomes_int = parameters["chromosomes"]
        chromosomes_str_long = []
        chromosomes_str_short = []
        for chromosome in parameters["chromosomes"]:
            chromosomes_str_long.append("chr" + str(chromosome))
            chromosomes_str_short.append(str(chromosome))

    return chromsomes_int, chromosomes_str_long, chromosomes_str_short

def load_predicted_tad_dict_per_methods(parameters):
    """
    Loads the dicts containing the tad prediction information for each method, and combines them into one final dict
    :param parameters: from json file: requires the "paths_predicted_tads_per_tad_prediction_methods"
    :return: tad_dict: dict containing all the tad prediction information for the methods given
    """
    paths = parameters["paths_predicted_tads_per_tad_prediction_methods"]

    tad_dict = {}
    for path in paths:
        with open(path, 'rb') as handle:
            method_dict = pickle.load(handle)
        tad_dict.update(method_dict)

    return tad_dict


def flatten_tad_dict(tad_dict):
    """
    flattens the list of tad regions in a tad dict
    :param tad_dict: tad dict {method: {chr: tads, chr tads}}
    :return: flat_tad_dict: same dict, but tad keys are now just lists instead of lists of lists
    """
    flat_tad_dict = dict.fromkeys(tad_dict.keys())
    for method in tad_dict:
        flat_tad_dict[method] = dict.fromkeys(tad_dict[method].keys())
        for chromosome in tad_dict[method]:
            if len(tad_dict[method][chromosome]) > 1:
                flat_tad_dict[method][chromosome] = [item for sublist in tad_dict[method][chromosome] for item in sublist]
            elif len(tad_dict[method][chromosome]) == 1:
                flat_tad_dict[method][chromosome] = [item for sublist in tad_dict[method][chromosome][0] for item in sublist]
            else:
                continue
    return flat_tad_dict


def jaccard_index_from_tad_dict(parameters, tad_prediction_methods, flat_tad_dict):
    """
    calculates jaccard index from tad sizes from different methods for corresponding chromosomes
    :param tad_prediction_methods: the methods used to predict tads in list form
    :param flat_tad_dict: dict of flattened tads for multiple methods for each chromosome
    :return: jaccard_index_tad_prediction_methods_combinations
    """

    jaccard_index_tad_prediction_methods_combinations = {}

    for tad_prediction_methods_combination in list(combinations(tad_prediction_methods, 2)):
        jaccard_index_chrom_list = []
        for chromosome in parameters["chromosomes_str_short"]:
            if (chromosome in flat_tad_dict[tad_prediction_methods_combination[0]]) & (chromosome in flat_tad_dict[tad_prediction_methods_combination[1]]):
                tads1 = flat_tad_dict[tad_prediction_methods_combination[0]][chromosome]
                tads2 = flat_tad_dict[tad_prediction_methods_combination[1]][chromosome]
                try:
                    jaccard_index = len(set(tads1) & set(tads2)) / len(set(tads1) | set(tads2))
                    jaccard_index_chrom_list.append(jaccard_index)
                except TypeError:
                    jaccard_index_chrom_list.append(np.nan)
        jaccard_index_tad_prediction_methods_combinations[tad_prediction_methods_combination] = jaccard_index_chrom_list

    # print(jaccard_index)
    print(jaccard_index_tad_prediction_methods_combinations)

    return jaccard_index_tad_prediction_methods_combinations


def venn_diagram_visualization(parameters, tad_prediction_methods, predicted_tads_per_tad_prediction_methods_chrom, chrnum):
    '''
    Function creates Venn Diagrams using genomic bins assigned with TAD/ No-TAD (called by two different TAD prediction methods) for each chromosome.

    :param parameters: dictionary with parameters set in parameters.json file
    :param tad_prediction_methods: list with used methods
    :param predicted_tads_per_tad_prediction_methods_chrom: list with flattened tad regions in the order of method in
    tad_prediction_methods list for the same chromosome
    :param chrnum: string of chromosome number for path name
    '''

    if len(tad_prediction_methods) == 2:

        venn2([set(predicted_tads_per_tad_prediction_methods_chrom[0]), set(predicted_tads_per_tad_prediction_methods_chrom[1])],
              set_labels=(tad_prediction_methods[0], tad_prediction_methods[1]), alpha=0.75)
        venn2_circles(
            [set(predicted_tads_per_tad_prediction_methods_chrom[0]), set(predicted_tads_per_tad_prediction_methods_chrom[1])],
            lw=0.7)

        # plt.show()

    elif len(tad_prediction_methods) == 3:

        venn3([set(predicted_tads_per_tad_prediction_methods_chrom[0]), set(predicted_tads_per_tad_prediction_methods_chrom[1]),
               set(predicted_tads_per_tad_prediction_methods_chrom[2])],
              set_labels=(tad_prediction_methods[0], tad_prediction_methods[1], tad_prediction_methods[2]), alpha=0.75)
        venn3_circles(
            [set(predicted_tads_per_tad_prediction_methods_chrom[0]), set(predicted_tads_per_tad_prediction_methods_chrom[1]),
             set(predicted_tads_per_tad_prediction_methods_chrom[2])], lw=0.7)
        # plt.show()

    elif len(tad_prediction_methods) > 3:

        sets = {}

        for tad_prediction_method, predicted_tads in zip(tad_prediction_methods,
                                                         predicted_tads_per_tad_prediction_methods_chrom):
            sets[tad_prediction_method] = set(predicted_tads_per_tad_prediction_methods_chrom)
            fig, ax = plt.subplots(1, figsize=(16, 12))
            venn(sets, ax=ax)
            plt.legend(tad_prediction_methods, ncol=6)
            # plt.show()
    else:
        raise ValueError("Only the predicted TADs for one method has been provided. No Venn diagram can be created.")

    plt.savefig(os.path.join(parameters["output_directory"], "evaluation", "venn_diagrams",
                             "Venn_diagram_of_" + "_" + chrnum + "_" + "_".join(tad_prediction_methods) + ".png"))
    plt.close()


def genomic_annotations_histogram(parameters, X, labels):
    """

    :param parameters: dictionary with parameters set in parameters.json file
    :param genomic_feature:
    :param X:
    :param labels:
    :return:
    """

    for index, genomic_feature in enumerate(parameters["node_feature_encoding"]):
        X_tad = X[labels == 1]
        X_notad = X[labels == 0]

        distribution_genomic_feature_X_tad = X_tad.T[index].flatten()
        distribution_genomic_feature_X_notad = X_notad.T[index].flatten()

        plt.hist(distribution_genomic_feature_X_tad, 50, alpha=0.5, label='x')
        plt.hist(distribution_genomic_feature_X_notad, 50, alpha=0.5, label='y')
        plt.legend(loc='upper right')
        # plt.show()
        plt.savefig(os.path.join(parameters["output_directory"], "evaluation", "venn_diagrams",
                                 "Distribution_of_" + genomic_feature + "for_tad_and_notad_genomic_bins.png"))
        plt.close()


def tad_region_size_calculation(parameters, predicted_tads_per_tad_prediction_methods_dict):
    """
    calculates the size of tad regions for each method
    :param parameters: json file
    :param predicted_tads_per_tad_prediction_methods_dict: dict containing methods and the tad regions detected for each chromosome
    :return:
    """

    for method in parameters["tad_prediction_methods"]:

        # predicted_tads_per_chromosomes = predicted_tads_per_tad_prediction_methods[method]
        # tad_region_chromosomes_dict = dict.fromkeys(parameters["cell_lines"])

        # for cell_line in parameters["cell_lines"]:
        tad_region_chromosomes_dict = dict.fromkeys(
            parameters["chromosomes_str_short"] + ["mean", "max", "min", "01quantile", "02quantile", "03quantile",
                                                   "04quantile", "05quantile", "06quantile", "07quantile", "08quantile",
                                                   "09quantile"])  # More or less assumes, that X is always present
        del tad_region_chromosomes_dict["X"]


        # for predicted_tads_per_method in predicted_tads_per_tad_prediction_methods[method]:
        tad_regions_across_chromosome = []
        for chromosome in predicted_tads_per_tad_prediction_methods_dict[method]:
            try:
                predicted_tads_per_method_chromosome = predicted_tads_per_tad_prediction_methods_dict[method][chromosome] # [0]
                tad_regions_within_chromosome = []
                tad_region_chromosomes_dict[chromosome] = dict.fromkeys(['mean_tad_size', 'number_tads'])
                tad_region_chromosomes_dict[chromosome]['number_tads'] = len(predicted_tads_per_method_chromosome)
                for tad in predicted_tads_per_method_chromosome:
                    tad_regions_across_chromosome.append(len(tad))
                    tad_regions_within_chromosome.append(len(tad))
            except IndexError:
                continue

            tad_region_chromosomes_dict[chromosome]['mean_tad_size'] = np.mean(tad_regions_within_chromosome)


        tad_region_chromosomes_dict["mean"] = np.mean(tad_regions_across_chromosome)
        tad_region_chromosomes_dict["max"] = np.mean(tad_regions_across_chromosome)
        tad_region_chromosomes_dict["min"] = np.mean(tad_regions_across_chromosome)
        tad_region_chromosomes_dict["01quantile"] = np.quantile(tad_regions_across_chromosome, 0.1)
        tad_region_chromosomes_dict["02quantile"] = np.quantile(tad_regions_across_chromosome, 0.2)
        tad_region_chromosomes_dict["03quantile"] = np.quantile(tad_regions_across_chromosome, 0.3)
        tad_region_chromosomes_dict["04quantile"] = np.quantile(tad_regions_across_chromosome, 0.4)
        tad_region_chromosomes_dict["05quantile"] = np.quantile(tad_regions_across_chromosome, 0.5)
        tad_region_chromosomes_dict["06quantile"] = np.quantile(tad_regions_across_chromosome, 0.6)
        tad_region_chromosomes_dict["07quantile"] = np.quantile(tad_regions_across_chromosome, 0.7)
        tad_region_chromosomes_dict["08quantile"] = np.quantile(tad_regions_across_chromosome, 0.8)
        tad_region_chromosomes_dict["09quantile"] = np.quantile(tad_regions_across_chromosome, 0.9)

        # print(tad_region_chromosomes_dict)

        tad_region_size_across_chromosome_visualisation(parameters, method, tad_regions_across_chromosome)

        output_path = os.path.join(parameters["output_directory"], "evaluation/", "tad_regions/",
                                   "TAD_region_size_statistics_with_" + method)
        print(output_path)
        with open(output_path, 'wb') as handle:
            pickle.dump(tad_region_chromosomes_dict, handle, protocol=pickle.DEFAULT_PROTOCOL)


def tad_region_size_across_chromosome_visualisation(parameters, tad_prediction_method, tad_regions_across_chromosome):
    plt.hist(tad_regions_across_chromosome, 50)
    plt.title(
        "TAD region size distribution across chromosomes for TAD prediction with " + tad_prediction_method)  ########################
    plt.xlabel("TAD region size")
    plt.ylabel("Number of TAD regions with the size.")
    # plt.show()
    plt.savefig(os.path.join(parameters["output_directory"], "evaluation/", "tad_regions/",
                             "TAD_region_size_distribution_" + tad_prediction_method))  ########################
    plt.close()


def create_adj_matrix(chr_len, tad_regions):
    adj = np.zeros((chr_len, chr_len))
    for tad in tad_regions: # [1, 2, 3]
        for combination in itertools.combinations_with_replacement(tad, 2): #[(1,1), (1,2) ...]
            adj[combination[0], combination[1]] = 1
            adj[combination[1], combination[0]] = 1
    return adj


def bed_to_dict_file(chr_str_short, bedfolder, chrorderlist, method, cell_line, scaling_factor, output_folder):
    """
    turns a .bed file, like the one produced by topdom into a dict containing the tad information
    :param chr_str_short: list of chr nums in string form
    :param bedfolder: folder where the bed files are contained
    :param chrorderlist: list (as a string) in the order of the chromosomes in the folder
    :param method: string of tad prediction method name
    :param cell_line: string of cell line used
    :param scaling_factor: int, either 100000 or 25000
    :param output_folder: folder in which the dict will be saved
    :return:
    """
    tad_dict = {}
    tad_dict[method] = dict.fromkeys(chr_str_short)

    bedfiles = []
    for file in os.listdir(bedfolder):
        if file.endswith(".bed"):
            bedfiles.append(os.path.join(bedfolder, file))

    for count, path in enumerate(bedfiles):
        df_tad_chr = pd.read_csv(path, delimiter="\t", header=None)

        df_tad_chr.iloc[:, 1] = df_tad_chr.iloc[:, 1].apply(lambda x: np.int(round(x / scaling_factor, 0)))
        df_tad_chr.iloc[:, 2] = df_tad_chr.iloc[:, 2].apply(lambda x: np.int(round(x / scaling_factor, 0)))

        df_tad_chr = df_tad_chr.rename(columns={0: "chr", 1: "x1", 2: "x2", 3: "label"})

        df_tad_chr = df_tad_chr.astype({"x1": 'int', "x2": 'int'})
        df_td = df_tad_chr[df_tad_chr["label"] == "domain"]
        td_solution_list = []
        for row in range(0, len(df_td)):
            r = range(df_td.iloc[row, 1], df_td.iloc[row, 2]) #not inclusive end pos
            # print(df_td.iloc[row,1], df_td.iloc[row, 2])
            l = [*r]
            # print(l)
            td_solution_list.append(l)
        td_solution_list.sort()
        tad_dict[method][chrorderlist[count]] = td_solution_list

    filepath = output_folder + method + "_" + cell_line + "_" + str(round(scaling_factor/1000)) + "kb_dict.p"

    with open(filepath, 'wb') as handle:
        pickle.dump(tad_dict, handle)

    return tad_dict


## OLD NPY FUNCTIONS
def pred_tads_to_dict(parameters, predicted_tads_per_method):
    """
    turns numpy array containing list of chromosomes with corresponding tad regions into a dict

    :param parameters: from json file
    :param predicted_tads_per_method: list of numpy arrays
    :return: tad_dict: {method: {chrom: tad, chrom:tad}}
    """
    predicted_tads_per_method_dict = dict.fromkeys(parameters["tad_prediction_methods"])
    for count, method in enumerate(parameters["tad_prediction_methods"]):
        predicted_tads_per_method_dict[method] = dict.fromkeys(parameters["chromosomes_str_short"])
        predicted_tads_chrom = predicted_tads_per_method[count]
        for chrom in parameters["chromosomes_str_short"]:
            try:
                idx_tad = np.where(predicted_tads_chrom == "chr" + chrom)
                idx_tad = idx_tad[0]
                predicted_tads_per_method_dict[method][chrom] = predicted_tads_chrom[idx_tad + 1]
            except ValueError:
                continue
    return predicted_tads_per_method_dict


def translate_bed_to_npy(parameters):
    """
    turns bed file with start bin, end bin and label columns into a npy array
    :param parameters:  json file with path to bed file for a specific chromosome
    :return:
    """
    file_TopDomSol = parameters["path_topdom_bed"]
    df_topdom = pd.read_csv(file_TopDomSol, delimiter="\t", header=None)
    resolution = 100000  # add to parameters?
    df_topdom.iloc[:, 1] = df_topdom.iloc[:, 1].apply(lambda x: np.int(round(x / resolution, 0)))
    df_topdom.iloc[:, 2] = df_topdom.iloc[:, 2].apply(lambda x: np.int(round(x / resolution, 0)))
    df_topdom = df_topdom.rename(columns={0: "chr", 1: "x1", 2: "x2", 3: "label"})
    df_topdom = df_topdom.astype({"x1": 'int', "x2": 'int'})

    df_td = df_topdom[df_topdom["label"] != "boundary"]
    td_solution_list = []
    for row in range(0, len(df_td)):
        r = range(df_td.iloc[row, 1], df_td.iloc[row, 2])
        l = [*r]
        td_solution_list.append(l)

    # np.save("/Users/Charlotte/VSC/MeetEU/tad_detection/evaluation/results/TopDom.npy", td_solution_list)


def load_predicted_tads_per_tad_prediction_methods(parameters):
    '''
    Function loads the predicted TADs of all methods provided in parameters.

    :param parameters: dictionary with parameters set in parameters.json file
    :return predicted_tads_per_tad_prediction_methods: array of predicted TADs separated in chromosomes and prediction method
    '''

    predicted_tads_per_tad_prediction_methods = []

    for path in parameters["paths_predicted_tads_per_tad_prediction_methods"]:
        all_predicted_tads = np.load(path, allow_pickle=True)
        for count in range(0, len(all_predicted_tads) - 1, 2):
            poss_cell_line = all_predicted_tads[count]
            if poss_cell_line == parameters["cell_line"]:
                predicted_tads_per_tad_prediction_methods.append(all_predicted_tads[count + 1])
            else:
                continue

    return predicted_tads_per_tad_prediction_methods

def flatten_tads_per_tad_prediction_methods(tad_predictions_per_tad_prediction_method):
    """
    tad regions in the form of list of lists [[1 2 3][5 6 7]] are flattened to a single list of regions [1 2 3 5 6 7]
    :param tad_predictions_per_tad_prediction_method: list of np arrays
    :return: flat_set_list: list of flattened np arrays
    """
    flat_set_list = []
    for method in tad_predictions_per_tad_prediction_method:
        method_flat_list = []
        for count in range(1, len(method), 2):
            flat_set = set([item for sub in method[count] for item in sub])
            method_flat_list.append([method[count - 1], flat_set])
        flat_set_list.append(method_flat_list)

    return flat_set_list


def align_chromosomes(parameters, flat_set_list_per_method):
    """
    Ensures that the tad ouputs from multiple chromosomes in two different methods are aligned.
    The order of the resulting list
    :param parameters: from parameters.json, used to access "chromosomes_int"
    :param flat_set_list_per_method: output from flatten_tads_per_prediction_methods
    :return: paired chromosomes: list containing two lists one for each method. In each of the method lists,
    there is another list with the flattened tad bins for the measured chromosomes. the order of the chromosomes for the
    different method align.
    paired_idx: list of chromosomes that are paired --> gives the order of the chromosomes
    """
    index_list_chr = []
    for methodnum in range(len(flat_set_list_per_method)):
        # print(methodnum)
        method_flat_set_list = flat_set_list_per_method[methodnum]
        order = [sub_list[0] for sub_list in method_flat_set_list]
        index_list_one_method = []
        for num in parameters["chromosomes_int"]:
            try:
                # print("chr", num)
                index = order.index("chr" + str(num))
                # print(index)
                index_list_one_method.append(index)
            except ValueError:
                index_list_one_method.append("NaN")
        index_list_chr.append(index_list_one_method)

    paired_chromosomes = []
    paired_idx = []
    for index in list(range(len(index_list_chr[0]))):
        try:
            index_1 = index_list_chr[0][index]
            index_2 = index_list_chr[1][index]
            paired_chrom = [flat_set_list_per_method[0][index_1], flat_set_list_per_method[1][index_2]]
            paired_chromosomes.append(paired_chrom)
            paired_idx.append(flat_set_list_per_method[0][index_1][0])
            # print(flat_set_list_per_method[0][index_1][0])
            # print(flat_set_list_per_method[1][index_2][0])
        except TypeError:
            continue
    return paired_chromosomes, paired_idx

def jaccard_index(tad_prediction_methods, aligned_flat_tads_per_method, paired_idx):
    '''
    Function calculates Jaccard index between genomic bins assigned with TAD/ No-TAD (called by two different TAD prediction methods) for each chromosome.

    :param tad_prediction_methods: list of TAD prediction methods for which TADs can be found in predicted_tads_per_tad_prediction_methods
    :param predicted_tads_per_tad_prediction_methods: array of predicted TADs separated in chromosomes and prediction method
    :return jaccard_index_tad_prediction_methods_combinations: list of Jaccard index between TADs called by two different methods for each chromosome
    '''

    jaccard_index_tad_prediction_methods_combinations = {}

    for tad_prediction_methods_combination in list(combinations(tad_prediction_methods, 2)):
        arr = np.where(np.array(tad_prediction_methods) == tad_prediction_methods_combination[0])[0]
        arr = arr.astype(int)
        arr = arr[0]
        print(arr)

        arr2 = np.where(np.array(tad_prediction_methods) == tad_prediction_methods_combination[1])[0]
        arr2 = arr2.astype(int)
        arr2 = arr2[0]
        print(arr2)

        jaccard_index_chrom_list = []
        for chrom in list(range(0, len(paired_idx))):
            tads1 = aligned_flat_tads_per_method[chrom][arr][1]
            tads2 = aligned_flat_tads_per_method[chrom][arr2][1]
            jaccard_index = len(set(tads1) & set(tads2)) / len(set(tads1) | set(tads2))
            jaccard_index_chrom_list.append(jaccard_index)
        jaccard_index_tad_prediction_methods_combinations[tad_prediction_methods_combination] = jaccard_index_chrom_list

    # print(jaccard_index)
    print(jaccard_index_tad_prediction_methods_combinations)

    return jaccard_index_tad_prediction_methods_combinations