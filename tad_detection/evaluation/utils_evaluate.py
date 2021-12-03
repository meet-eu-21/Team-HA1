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
        predicted_tads = np.load(path)
        predicted_tads_per_tad_prediction_methods.append(predicted_tads)

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

def jaccard_index_weighted(parameters, tad_prediction_methods, predicted_tads_per_tad_prediction_methods):

    #We may need to discuss whether we are able to weigh our TAD predictions, e.g. by the number of annotations with CTCF?

    return 0

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