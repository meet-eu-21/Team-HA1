import sys
sys.path.insert(1, './tad_detection/')
sys.path.insert(1, './tad_detection/preprocessing/')
sys.path.insert(1, './tad_detection/model/')
sys.path.insert(1, './tad_detection/evaluation/')

import pandas as pd
import numpy as np
import os
import argparse
from utils_preprocessing import generate_chromosome_lists
from utils_general import load_parameters, set_up_logger
import pyBigWig
import gffutils
import pickle
import math

def generate_dict_genomic_annotations(parameters, cell_line):
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
            annot_CTCF.append(np.sum(file_CTCF.values(chromosome, gbins[-1], math.floor(chr_len/ 100) * 100)))
            annotations.append(annot_CTCF)
        if 'RAD21' in parameters["genomic_annotations"]:
            annot_RAD21 = [np.sum(file_RAD21.values(chromosome, gbin, (gbin + (parameters["scaling_factor"] - 1))))
                           for gbin in gbins[:-1]]
            annot_RAD21.append(np.sum(file_RAD21.values(chromosome, gbins[-1], math.floor(chr_len/ 100) * 100)))
            annotations.append(annot_RAD21)
        if 'SMC3' in parameters["genomic_annotations"]:
            annot_SMC3 = [np.sum(file_SMC3.values(chromosome, gbin, (gbin + (parameters["scaling_factor"] - 1))))
                          for gbin in gbins[:-1]]
            annot_SMC3.append(np.sum(file_SMC3.values(chromosome, gbins[-1], math.floor(chr_len/ 100) * 100)))
            annotations.append(annot_SMC3)

        values = list(zip(*annotations))

        positions = [(x + 1) for x in range(int(math.ceil(chr_len / parameters["scaling_factor"])))]
        dict_values.append(dict(zip(positions, values)))
    dict_target_genes = dict(zip(chromosomes_str_short, dict_values))
    with open(os.path.join(parameters["genomic_annotations_dicts_directory"], "genomic_annotations_" + cell_line + "_" + parameters["resolution_hic_matrix_string"] + ".pickle"), 'wb') as handle:
        pickle.dump(dict_target_genes, handle, protocol=pickle.DEFAULT_PROTOCOL)


def load_dict_genomic_annotations(parameters, cell_line):
    with open(os.path.join(parameters["genomic_annotations_dicts_directory"], "genomic_annotations_" + cell_line +
                           "_" + parameters["resolution_hic_matrix_string"] + ".pickle"), 'rb') as handle:
        dict_target_genes = pickle.load(handle)

    return dict_target_genes


def load_housekeeping_genes(parameters):

    housekeeping_genes = pd.read_csv("./ressources/Housekeeping_GenesHuman.csv", delimiter=";")

    housekeeping_genes = list(housekeeping_genes["Gene.name"])

    return housekeeping_genes


def ensembl_gtf_database(parameters):

    if not os.path.isfile(parameters["ensembl_gtf_database_path"]):
        fn = gffutils.example_filename(parameters["ensembl_gtf_file_path"])
        gtf = gffutils.create_db(fn, dbfn=parameters["ensembl_gtf_database_path"], force=True, keep_order=True, merge_strategy='merge', sort_attribute_values=True)
    gtf = gffutils.FeatureDB("/data/analysis/ag-reils/ag-reils-shared/Hi-C/meeteu/ensembl/Homo_sapiens.GRCh37.55.db",
                            keep_order=True)

    return gtf


def generate_dict_housekeeping_genes(housekeeping_genes, gtf, parameters):

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
    with open(os.path.join(parameters["genomic_annotations_dicts_directory"], "housekeeping_genes_" + parameters["resolution_hic_matrix_string"] + ".pickle"), 'rb') as handle:
        dict_housekeeping_genes = pickle.load(handle)

    return dict_housekeeping_genes


def combine_genomic_annotations_and_housekeeping_genes(parameters, arrowhead_solution_list,
                                                       dict_genomic_annotations=None, dict_housekeeping_genes=None):

    number_bins_adjacency_matrices_arrowhead_solution = []
    chr_data = pd.read_csv("../ressources/hg19_chrom_sizes.txt", sep='\t', header=None)
    chromosomes = chr_data[0].tolist()
    chr_lengths = chr_data[1].tolist()
    for cell_line in range(len(parameters["cell_lines"])):
        number_bins_adjacency_matrices_arrowhead_solution_cell_line = []
        for chromosome in range(len(arrowhead_solution_list[cell_line])):
            number_bins_adjacency_matrices_arrowhead_solution_cell_line.append(len(arrowhead_solution_list[cell_line][chromosome]))

        number_bins_adjacency_matrices_arrowhead_solution.append(number_bins_adjacency_matrices_arrowhead_solution_cell_line)

    annotation_matrices_list_cell_lines = []
    for index_cell_line, cell_line in enumerate(parameters["cell_lines"]):
        annotation_matrices_list = []
        for index_chromosome, chromosome in enumerate(chromosomes):
            chr_len = chr_lengths[index_chromosome]
            bins = [(x + 1) for x in range(int(math.ceil(chr_len / parameters["scaling_factor"])))]
            annotation_matrix = np.zeros((len(bins), len(parameters["genomic_annotations"])))

            if 'housekeeping_genes' in parameters["genomic_annotations"]:
                if ('CTCF' or 'RAD21' or 'SMC3') in parameters["genomic_annotations"]:
                    annotation_matrix[[(x - 1) for x in list(dict_genomic_annotations[cell_line][chromosome].keys())],
                    :(len(parameters["genomic_annotations"]) - 1)] = list(
                        dict_genomic_annotations[cell_line][chromosome].values())
                for bin in dict_housekeeping_genes[chromosome].keys():
                    annotation_matrix[(bin - 1), (len(parameters["genomic_annotations"]) - 1)] = \
                        dict_housekeeping_genes[chromosome][bin]
            elif ('CTCF' or 'RAD21' or 'SMC3') in parameters["genomic_annotations"]:
                annotation_matrix[[(x - 1) for x in list(dict_genomic_annotations[cell_line][chromosome].keys())],
                :(len(parameters["genomic_annotations"]))] = list(
                    dict_genomic_annotations[cell_line][chromosome].values())
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_parameters_json", help=" to JSON with parameters.", type=str, required=True)
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

