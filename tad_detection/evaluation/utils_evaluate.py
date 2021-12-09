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
from itertools import combinations
import pickle

def _old_load_parameters(path_parameters_json):
    '''
    Function loads the parameters from the provided parameters.json file in a dictionary.

    :param path_parameters_json: path of parameters.json file
    :return parameters: dictionary with parameters set in parameters.json file
    '''

    with open(path_parameters_json) as parameters_json:
        parameters = json.load(parameters_json)

    return parameters

def _old_set_up_logger(parameters):
    '''
    Function sets a global logger for documentation of information and errors in the execution of the chosen script.

    :param parameters: dictionary with parameters set in parameters.json file
    '''

    logger = logging.getLogger('evaluation')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(parameters["output_directory"], 'evaluation', 'evaluation.log'))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    logger.info("Evaluation with the following parameters:")
    for parameter in parameters.keys():
        logger.info(parameter + ": " + str(parameters[parameter]))

def load_predicted_tads_per_tad_prediction_methods(parameters):
    '''

    :param parameters: dictionary with parameters set in parameters.json file
    :return:
    '''

    predicted_tads_per_tad_prediction_methods = []

    for path in parameters["paths_predicted_tads_per_tad_prediction_methods"]:
        predicted_tads = np.load(path, allow_pickle=True)
        predicted_tads_per_tad_prediction_methods.append(predicted_tads)

    #TODO
    #Restrict loading of other solutions to chromosomes chosen
    #Only load one cell line at a time
    return predicted_tads_per_tad_prediction_methods

def jaccard_index(tad_prediction_methods, predicted_tads_per_tad_prediction_methods):
    '''

    :param tad_prediction_methods:
    :param predicted_tads_per_tad_prediction_methods:
    :return:
    '''

    jaccard_index_tad_prediction_methods_combinations = {}

    for tad_prediction_methods_combination in list(combinations(tad_prediction_methods, 2)):
        tads1 = predicted_tads_per_tad_prediction_methods[np.where(np.array(tad_prediction_methods) == tad_prediction_methods_combination[0])]
        tads2 = predicted_tads_per_tad_prediction_methods[np.where(np.array(tad_prediction_methods) == tad_prediction_methods_combination[1])]

        jaccard_index = len(set(tads1) & set(tads2)) / len(set(tads1) | set(tads2))
        jaccard_index_tad_prediction_methods_combinations[tad_prediction_methods_combination] = jaccard_index

    return jaccard_index_tad_prediction_methods_combinations

def venn_diagram_visualization(parameters, tad_prediction_methods, predicted_tads_per_tad_prediction_methods):
    '''


    :param parameters: dictionary with parameters set in parameters.json file
    :param tad_prediction_methods:
    :param predicted_tads_per_tad_prediction_methods:
    :return:
    '''

    if len(tad_prediction_methods) == 2:

        venn2([set(predicted_tads_per_tad_prediction_methods[0]), set(predicted_tads_per_tad_prediction_methods[1])], set_labels=(tad_prediction_methods[0], tad_prediction_methods[1]), alpha=0.75)
        venn2_circles([set(predicted_tads_per_tad_prediction_methods[0]), set(predicted_tads_per_tad_prediction_methods[1])], lw=0.7)
        #plt.show()

    elif len(tad_prediction_methods) == 3:

        venn3([set(predicted_tads_per_tad_prediction_methods[0]), set(predicted_tads_per_tad_prediction_methods[1]), set(predicted_tads_per_tad_prediction_methods[3])], set_labels=(tad_prediction_methods[0], tad_prediction_methods[1], tad_prediction_methods[2]), alpha=0.75)
        venn3_circles([set(predicted_tads_per_tad_prediction_methods[0]), set(predicted_tads_per_tad_prediction_methods[1]), set(predicted_tads_per_tad_prediction_methods[2])], lw=0.7)
        #plt.show()

    elif len(tad_prediction_methods) > 3:

        sets = {}

        for tad_prediction_method, predicted_tads in zip(tad_prediction_methods, predicted_tads_per_tad_prediction_methods):
            sets[tad_prediction_method] = set(predicted_tads_per_tad_prediction_methods)
            fig, ax = plt.subplots(1, figsize=(16, 12))
            venn(sets, ax=ax)
            plt.legend(tad_prediction_methods, ncol=6)
            #plt.show()
    else:
        raise ValueError("Only the predicted TADs for one method has been provided. No Venn diagram can be created.")

    plt.savefig(os.path.join(parameters["output_directory"], "evaluation", "venn_diagrams",
                             "Venn_diagram_of_" + "_".join(tad_prediction_methods) + ".png"))

def genomic_annotations_histogram(parameters, X, labels):
    '''

    :param parameters: dictionary with parameters set in parameters.json file
    :param genomic_feature:
    :param X:
    :param labels:
    :return:
    '''

    for index, genomic_feature in enumerate(parameters["node_feature_encoding"]):
        X_tad = X[labels == 1]
        X_notad = X[labels == 0]

        distribution_genomic_feature_X_tad = X_tad.T[index].flatten()
        distribution_genomic_feature_X_notad = X_notad.T[index].flatten()

        plt.hist(distribution_genomic_feature_X_tad, 50, alpha=0.5, label='x')
        plt.hist(distribution_genomic_feature_X_notad, 50, alpha=0.5, label='y')
        plt.legend(loc='upper right')
        #plt.show()
        plt.savefig(os.path.join(parameters["output_directory"], "evaluation", "venn_diagrams",
                                 "Distribution_of_" + genomic_feature + "for_tad_and_notad_genomic_bins.png"))

def generate_chromosome_lists(parameters):
    '''

    :param parameters: dictionary with parameters set in parameters.json file
    :return:
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

def tad_region_size_calculation(parameters, predicted_tads_per_tad_prediction_methods):

    #TODO
    #Remove cell line stuff, because we only import for a specific cell line

    for tad_prediction_method, predicted_tads_per_tad_prediction_method in zip(parameters["tad_prediction_methods"], predicted_tads_per_tad_prediction_methods):

        tad_region_chromosomes_dict = dict.fromkeys(parameters["cell_lines"])

        for cell_line in parameters["cell_lines"]:
            tad_region_chromosomes_dict[cell_line] = dict.fromkeys(parameters["chromsomes_int"] + [23, "mean", "max", "min", "01quantile", "02quantile", "03quantile", "04quantile", "05quantile", "06quantile", "07quantile", "08quantile", "09quantile"]) #More or less assumes, that X is always present
            del tad_region_chromosomes_dict[cell_line]["X"]

        for cell_line, predicted_tads_per_tad_prediction_method_cell_line in zip(parameters["cell_lines"], predicted_tads_per_tad_prediction_method):
            tad_regions_across_chromosome = []
            for index_chromosome, predicted_tads_per_tad_prediction_method_cell_line_chromosome in enumerate(predicted_tads_per_tad_prediction_method_cell_line):
                tad_regions_within_chromosome = []
                for tad in predicted_tads_per_tad_prediction_method_cell_line_chromosome:
                    tad_regions_across_chromosome.append(len(tad))
                    tad_regions_within_chromosome.append(len(tad))
                tad_region_chromosomes_dict[cell_line][index_chromosome] = np.mean(tad_regions_within_chromosome)
            tad_region_chromosomes_dict[cell_line]["mean"] = np.mean(tad_regions_across_chromosome)
            tad_region_chromosomes_dict[cell_line]["max"] = np.mean(tad_regions_across_chromosome)
            tad_region_chromosomes_dict[cell_line]["min"] = np.mean(tad_regions_across_chromosome)
            tad_region_chromosomes_dict[cell_line]["01quantile"] = np.quantile(tad_regions_across_chromosome, 0.1)
            tad_region_chromosomes_dict[cell_line]["02quantile"] = np.quantile(tad_regions_across_chromosome, 0.2)
            tad_region_chromosomes_dict[cell_line]["03quantile"] = np.quantile(tad_regions_across_chromosome, 0.3)
            tad_region_chromosomes_dict[cell_line]["04quantile"] = np.quantile(tad_regions_across_chromosome, 0.4)
            tad_region_chromosomes_dict[cell_line]["05quantile"] = np.quantile(tad_regions_across_chromosome, 0.5)
            tad_region_chromosomes_dict[cell_line]["06quantile"] = np.quantile(tad_regions_across_chromosome, 0.6)
            tad_region_chromosomes_dict[cell_line]["07quantile"] = np.quantile(tad_regions_across_chromosome, 0.7)
            tad_region_chromosomes_dict[cell_line]["08quantile"] = np.quantile(tad_regions_across_chromosome, 0.8)
            tad_region_chromosomes_dict[cell_line]["09quantile"] = np.quantile(tad_regions_across_chromosome, 0.9)
            tad_region_size_across_chromosome_visualisation(parameters, cell_line, tad_regions_across_chromosome)

        with open(os.path.join(parameters["output_directory"], "/evaluation/", "/tad_regions/", "TAD_region_size_statistics_with" + tad_prediction_method), 'wb') as handle:
            pickle.dump(tad_region_chromosomes_dict, handle, protocol=pickle.DEFAULT_PROTOCOL)

def tad_region_size_across_chromosome_visualisation(parameters, cell_line, tad_prediction_method, tad_regions_across_chromosome):

    plt.hist(tad_regions_across_chromosome, 50)
    plt.title("TAD region size distribution across chromosomes of cell line " + cell_line + "for TAD prediction with " + tad_prediction_method) ########################
    plt.xlabel("TAD region size")
    plt.ylabel("Number of TAD regions with the size.")
    #plt.show()
    plt.save(os.path.join(parameters["output_directory"], "/evaluation/", "/tad_regions/", "TAD_region_size_distribution_" + cell_line + "_" + tad_prediction_method)) ########################

'''
def jaccard_index_per_chromosome(tad_prediction_methods, predicted_tads_per_tad_prediction_methods):

    jaccard_index_per_chromosome = jaccard_index(tad_prediction_methods, predicted_tads_per_tad_prediction_methods)

    return jaccard_index_per_chromosome

def jaccard_index_all_chromosomes(tad_prediction_methods, predicted_tads_per_tad_prediction_methods):

    #TODO
    #REmove cell line stuff as in tad_region_size

    for predicted_tads_per_tad_prediction_method in predicted_tads_per_tad_prediction_methods:
        for predicted_tads_per_tad_prediction_method_cell_line in zip(predicted_tads_per_tad_prediction_method):
            genomic_bins_tad_cell_line = []
            for predicted_tads_per_tad_prediction_method_cell_line_chromosome in predicted_tads_per_tad_prediction_method_cell_line:
                for tad in predicted_tads_per_tad_prediction_method_cell_line_chromosome:
                    for bins in tad:
                        print("sfdghjalkhg")
                        #genomic_bins_tad.append(bin)

    jaccard_index_all_chromosomes = jaccard_index(tad_prediction_methods, predicted_tads_per_tad_prediction_methods)

    return jaccard_index_all_chromosomes

def venn_diagram_visualization_per_chromosome(parameters, tad_prediction_methods, predicted_tads_per_tad_prediction_methods):
    #TODO

def venn_diagram_visualization_all_chromosomes(parameters, tad_prediction_methods, predicted_tads_per_tad_prediction_methods):
    #TODO
'''