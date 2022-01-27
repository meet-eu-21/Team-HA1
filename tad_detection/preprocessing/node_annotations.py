import sys
sys.path.insert(1, './tad_detection/')
sys.path.insert(1, './tad_detection/preprocessing/')
sys.path.insert(1, './tad_detection/model/')
sys.path.insert(1, './tad_detection/evaluation/')

import logging
logger = logging.getLogger('node_annotations')

import pandas as pd
import numpy as np
import os
import argparse
from utils_general import load_parameters, set_up_logger
from utils_preprocessing import generate_chromosome_lists
import pyBigWig
import gffutils
import pickle
import math


def generate_dict_genomic_annotations(parameters, cell_line):
    """
    Function generates and saves lists of the number of annotations for different genomic annotations in each genomic
    bin (for each chromsome). If binary is set to True, the generated lists are one-hot encoded, if the sum of
    annotations in a certain bin reaches a given threshold (specified quantiles).

    :param parameters: dictionary with parameters set in parameters.json file (If `binary_dict` is false, the sum of
    annotations is saved for each bin. If `binary_dict` is true, a one-hot encoded list is generated.)
    :param cell_line: Cell line (GM12878 or IMR-90)
    """

    _, chromosomes_str_long, chromosomes_str_short = generate_chromosome_lists(parameters)
    dict_values = []
    if cell_line == 'GM12878':
        file_for_chr_len = pyBigWig.open(
            parameters[f"genomic_annotations_{parameters['genomic_annotations'][0]}_GM12878"])
        if 'CTCF' in parameters["genomic_annotations"]:
            file_CTCF = pyBigWig.open(parameters["genomic_annotations_CTCF_GM12878"])
        if 'RAD21' in parameters["genomic_annotations"]:
            file_RAD21 = pyBigWig.open(parameters["genomic_annotations_RAD21_GM12878"])
        if 'SMC3' in parameters["genomic_annotations"]:
            file_SMC3 = pyBigWig.open(parameters["genomic_annotations_SMC3_GM12878"])
    elif cell_line == 'IMR-90':
        file_for_chr_len = pyBigWig.open(
            parameters[f"genomic_annotations_{parameters['genomic_annotations'][0]}_IMR-90"])
        if 'CTCF' in parameters["genomic_annotations"]:
            file_CTCF = pyBigWig.open(parameters["genomic_annotations_CTCF_IMR-90"])
        if 'RAD21' in parameters["genomic_annotations"]:
            file_RAD21 = pyBigWig.open(parameters["genomic_annotations_RAD21_IMR-90"])
        if 'SMC3' in parameters["genomic_annotations"]:
            file_SMC3 = pyBigWig.open(parameters["genomic_annotations_SMC3_IMR-90"])
    else:
        raise ValueError('Wrong cell line used, use one of GM12878 or IMR-90')

    for chromosome in chromosomes_str_long:
        chr_len = file_for_chr_len.chroms()[chromosome]

        gbins = [x for x in range(0, chr_len, parameters["scaling_factor"])]

        annotations = []
        if 'CTCF' in parameters["genomic_annotations"]:
            annot_CTCF = [np.sum(file_CTCF.values(chromosome, gbin, (gbin + (parameters["scaling_factor"] - 1))))
                          for gbin in gbins[:-1]]
            annot_CTCF.append(np.sum(file_CTCF.values(chromosome, gbins[-1], math.floor(chr_len / 100) * 100)))
            if parameters["binary_dict"]:
                annot_CTCF_binary = (
                            annot_CTCF > np.quantile(annot_CTCF, parameters["quantile_genomic_annotations"])).astype(
                    dtype=int)
                annotations.append(annot_CTCF_binary)
            else:
                annotations.append(annot_CTCF)
        if 'RAD21' in parameters["genomic_annotations"]:
            annot_RAD21 = [np.sum(file_RAD21.values(chromosome, gbin, (gbin + (parameters["scaling_factor"] - 1))))
                           for gbin in gbins[:-1]]
            annot_RAD21.append(np.sum(file_RAD21.values(chromosome, gbins[-1], math.floor(chr_len / 100) * 100)))
            if parameters["binary_dict"]:
                annot_RAD21_binary = (
                            annot_RAD21 > np.quantile(annot_RAD21, parameters["quantile_genomic_annotations"])).astype(
                    dtype=int)
                annotations.append(annot_RAD21_binary)
            else:
                annotations.append(annot_RAD21)
        if 'SMC3' in parameters["genomic_annotations"]:
            annot_SMC3 = [np.sum(file_SMC3.values(chromosome, gbin, (gbin + (parameters["scaling_factor"] - 1))))
                          for gbin in gbins[:-1]]
            annot_SMC3.append(np.sum(file_SMC3.values(chromosome, gbins[-1], math.floor(chr_len / 100) * 100)))
            if parameters["binary_dict"]:
                annot_SMC3_binary = (
                            annot_SMC3 > np.quantile(annot_SMC3, parameters["quantile_genomic_annotations"])).astype(
                    dtype=int)
                annotations.append(annot_SMC3_binary)
            annotations.append(annot_SMC3)

        values = list(zip(*annotations))

        positions = [(x + 1) for x in range(int(math.ceil(chr_len / parameters["scaling_factor"])))]
        dict_values.append(dict(zip(positions, values)))
    dict_target_genes = dict(zip(chromosomes_str_short, dict_values))
    if parameters["binary_dict"]:
        with open(os.path.join(parameters["genomic_annotations_dicts_directory"], "genomic_annotations_" + cell_line + "_" + parameters["resolution_hic_matrix_string"] + "_" + parameters["quantile_genomic_annotations"] + "quantile" + ".pickle"), 'wb') as handle:
            pickle.dump(dict_target_genes, handle, protocol=pickle.DEFAULT_PROTOCOL)
    else:
        with open(os.path.join(parameters["genomic_annotations_dicts_directory"], "genomic_annotations_" + cell_line + "_" + parameters["resolution_hic_matrix_string"] + ".pickle"), 'wb') as handle:
            pickle.dump(dict_target_genes, handle, protocol=pickle.DEFAULT_PROTOCOL)


def load_dict_genomic_annotations(parameters, cell_line):
    """
    Function loads dict of lists indicating the number of annotations of specific genomic annotations in each genomic
    bin for each chromsome.

    :param parameters: dictionary with parameters set in parameters.json file
    :param cell_line: Cell line (GM12878 or IMR-90)
    :return:
    """

    if parameters["binary_dict"] == "True":
        logger.info(f"Load binary genomic annotations dict with quantile {str(parameters['quantile_genomic_annotations'])}.")
        with open(os.path.join(parameters["genomic_annotations_dicts_directory"], "genomic_annotations_" + cell_line +
                            "_" + parameters["resolution_hic_matrix_string"] + "_" +
                            str(parameters["quantile_genomic_annotations"]) + "quantile" + ".pickle"), 'rb') as handle:
            dict_target_genes = pickle.load(handle)
    else:
        logger.info(f"Load genomic annotations dict.")
        with open(os.path.join(parameters["genomic_annotations_dicts_directory"], "genomic_annotations_" + cell_line +
                            "_" + parameters["resolution_hic_matrix_string"] + ".pickle"), 'rb') as handle:
            dict_target_genes = pickle.load(handle)

    return dict_target_genes


def load_housekeeping_genes(parameters):
    """
    Function loads a .csv with the housekeeping genes in humans.

    :param parameters: dictionary with parameters set in parameters.json file
    :return housekeeping_genes: list of housekeeping genes in humans
    """

    housekeeping_genes = pd.read_csv(parameters["path_housekeeping_genes"], delimiter=";")

    housekeeping_genes = list(housekeeping_genes["Gene.name"])

    return housekeeping_genes


def ensembl_gtf_database(parameters):
    """
    Function creates or re-loads a database from an ensembl gtf file with information on features (e.g. genes), positions in the sequence etc. Please refer to Ensembl: https://m.ensembl.org/info/website/upload/gff.html

    :param parameters: dictionary with parameters set in parameters.json file
    :return gtf: database with content from an esenbml gtf file
    """

    if not os.path.isfile(parameters["ensembl_gtf_database_path"]):
        fn = gffutils.example_filename(parameters["ensembl_gtf_file_path"])
        gtf = gffutils.create_db(fn, dbfn=parameters["ensembl_gtf_database_path"], force=True, keep_order=True, merge_strategy='merge', sort_attribute_values=True)
    gtf = gffutils.FeatureDB("/data/analysis/ag-reils/ag-reils-shared/Hi-C/meeteu/ensembl/Homo_sapiens.GRCh37.55.db",
                            keep_order=True)

    return gtf


def generate_dict_housekeeping_genes(housekeeping_genes, gtf, parameters):
    """
    Function generates and saves one-hot encoded lists of the presence of housekeeping genes in each genomic bin in each each adjacency matrix (for each chromsome).

    :param housekeeping_genes: list of housekeeping genes in humans
    :param gtf: database with content from an esenbml gtf file
    :param parameters: dictionary with parameters set in parameters.json file
    """

    dict_housekeeping_genes = {}

    for chromosome in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17",
                       "18", "19", "20", "21", "22", "X"]:
        dict_housekeeping_genes[chromosome] = {}

    ensemblgeneid_externalgeneid_translation = pd.read_csv(parameters["translation_ensembl_geneid_to_external_geneid"])

    for housekeeping_gene in housekeeping_genes:
        housekeeping_gene = list(ensemblgeneid_externalgeneid_translation[ensemblgeneid_externalgeneid_translation["external_gene_name"] == housekeeping_gene]["ensembl_gene_id"])
        if len(housekeeping_gene) == 1:
            housekeeping_gene = housekeeping_gene[0]
            try:
                housekeeping_gene = gtf[housekeeping_gene]
                bin_start = int(np.round(housekeeping_gene.start / parameters["scaling_factor"], 0))
                bin_end = int(np.round(housekeeping_gene.start / parameters["scaling_factor"], 0))
                for genomic_position in [bin_start, bin_end]:
                    if dict_housekeeping_genes[housekeeping_gene.chrom].get(genomic_position):
                        dict_housekeeping_genes[housekeeping_gene.chrom][genomic_position] += 1
                    else:
                        dict_housekeeping_genes[housekeeping_gene.chrom][genomic_position] = 1
                    if bin_start == bin_end:
                        break
            except:
                continue

    with open(os.path.join(parameters["genomic_annotations_dicts_directory"], "housekeeping_genes_" + parameters["resolution_hic_matrix_string"] + ".pickle"), 'wb') as handle:
        pickle.dump(dict_housekeeping_genes, handle, protocol=pickle.DEFAULT_PROTOCOL)


def load_dict_housekeeping_genes(parameters):
    """
    Function loads dict of one-hot encoded lists indicating the presence of housekeeping genes in each genomic bin for each chromsome.

    :param parameters: dictionary with parameters set in parameters.json file
    :return dict_housekeeping_gene: dict with one-hot encoded lists for each chromosome indicating the presenece of housekeeping genes in each genomic bin in a chromosome
    """

    with open(os.path.join(parameters["genomic_annotations_dicts_directory"], "housekeeping_genes_" + parameters["resolution_hic_matrix_string"] + ".pickle"), 'rb') as handle:
        dict_housekeeping_genes = pickle.load(handle)

    return dict_housekeeping_genes


def combine_genomic_annotations_and_housekeeping_genes(parameters, arrowhead_solution_list, adjacency_matrices_source_information_list, dict_genomic_annotations=None, dict_housekeeping_genes=None):
    """
    Combines the dict containing the one-hot encded lists for the housekeeping genes (dict_housekeeping_genes) and the
    dict containing the lists for the genomic annotations (dict_genomic_annotations). It generates annotation matrices
    for each cell line and each chromosome containing the features of nodes (housekeeping genes and genomic annotations.

    :param parameters: dictionary with parameters set in parameters.json file
    :param arrowhead_solution_list: TAD classification for genomic bins in original HiC-maps/ graph nodes called by Arrowhead method (Ground truth) separated for chromosomes and cell line
    :param adjacency_matrices_source_information_list: source information for adjacency_matrices_list indicating the chromosome of the corresponding adjacency matrix
    :param dict_genomic_annotations: dict with lists indicating the number of annotations of specific genomic annotations in each genomic bin for each chromsome.
    :param dict_housekeeping_genes: dict with one-hot encoded lists for each chromosome indicating the presence of housekeeping genes in each genomic bin in a chromosome
    :return annotation_matrices_list_cell_lines: features of nodes in annotation matrices separated for chromosomes and cell line
    """

    number_bins_adjacency_matrices_arrowhead_solution = []
    for cell_line in range(len(parameters["cell_lines"])):
        number_bins_adjacency_matrices_arrowhead_solution_cell_line = []
        for chromosome in range(len(arrowhead_solution_list[cell_line])):
            number_bins_adjacency_matrices_arrowhead_solution_cell_line.append(len(arrowhead_solution_list[cell_line][chromosome]))

        number_bins_adjacency_matrices_arrowhead_solution.append(number_bins_adjacency_matrices_arrowhead_solution_cell_line)

    annotation_matrices_list_cell_lines = []
    for index_cell_line, cell_line in enumerate(parameters["cell_lines"]):
        annotation_matrices_list = []
        for index_chromosome, (chr_len, chromosome) in enumerate(zip(number_bins_adjacency_matrices_arrowhead_solution[index_cell_line], adjacency_matrices_source_information_list[index_cell_line])):
            annotation_matrix = np.zeros((chr_len, len(parameters["genomic_annotations"])))
            chromosome = chromosome.rsplit('-', maxsplit=1)[1]

            if 'housekeeping_genes' in parameters["genomic_annotations"]:
                if ('CTCF' or 'RAD21' or 'SMC3') in parameters["genomic_annotations"]:
                    annotation_matrix[[(x - 1) for x in list(dict_genomic_annotations[cell_line][chromosome].keys())[:chr_len]],
                    :(len(parameters["genomic_annotations"]) - 1)] = list(
                        dict_genomic_annotations[cell_line][chromosome].values())[:chr_len]
                for bin in dict_housekeeping_genes[chromosome].keys():
                    annotation_matrix[(bin - 1), (len(parameters["genomic_annotations"]) - 1)] = \
                        dict_housekeeping_genes[chromosome][bin]
            elif ('CTCF' or 'RAD21' or 'SMC3') in parameters["genomic_annotations"]:
                annotation_matrix[[(x - 1) for x in list(dict_genomic_annotations[cell_line][chromosome].keys())[:chr_len]],
                :(len(parameters["genomic_annotations"]))] = list(
                    dict_genomic_annotations[cell_line][chromosome].values())[:chr_len]
            else:
                raise ValueError("No genomic annotations or housekeeping genes mentioned in the parameter 'genomic "
                                 "annotations', so no annotation matrices can be computed.")

            annotation_matrix = annotation_matrix[:number_bins_adjacency_matrices_arrowhead_solution[index_cell_line][index_chromosome], ]
            annotation_matrices_list.append(annotation_matrix)

        annotation_matrices_list_cell_lines.append(np.array(annotation_matrices_list))
        #np.save(os.path.join(parameters["genomic_annotations_dicts_directory"], "annotation_matrix_" + cell_line + "_" + parameters["resolution_hic_matrix_string"] + ".npy"),
        #        final_annotation_matrix)

    return annotation_matrices_list_cell_lines


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create dictionaries of genomic annotations and occurrence of housekeeping genes for genomic bins used in preprocessing pipeline used for dataset creation.')
    parser.add_argument("--path_parameters_json", help="path to JSON with parameters.", type=str, required=True)
    args = parser.parse_args()
    path_parameters_json = args.path_parameters_json
    #path_parameters_json = "./tad_detection/preprocessing/parameters.json"

    parameters = load_parameters(path_parameters_json)
    os.makedirs(parameters["output_directory"], exist_ok=True)
    os.makedirs(os.path.join(parameters["output_directory"], "node_annotations"), exist_ok=True)
    logger = set_up_logger('node_annotations', parameters)
    logger.debug('Start node_annotations logger.')

    for cell_line in parameters["cell_lines"]:
        if cell_line == "GM12878" or cell_line == "IMR-90":
            if not os.path.isfile(os.path.join(parameters["genomic_annotations_dicts_directory"], "genomic_annotations_" + cell_line + "_" + parameters["resolution_hic_matrix_string"] + ".pickle")):
                generate_dict_genomic_annotations(parameters, cell_line)
    if not os.path.isfile(os.path.join(parameters["genomic_annotations_dicts_directory"], "housekeeping_genes_" + parameters["resolution_hic_matrix_string"] + ".pickle")):
        housekeeping_genes = load_housekeeping_genes(parameters)
        gtf = ensembl_gtf_database(parameters)
        generate_dict_housekeeping_genes(housekeeping_genes, gtf, parameters)

