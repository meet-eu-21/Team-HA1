import pandas as pd
import numpy as np
import os
import argparse
from utils import load_parameters
import gffutils
import pickle

def load_genomic_annotations(parameters):
    return 0

def generate_dict_genomic_annotations():

    return 0

def load_dicts_genomic_annotations(parameters):

    parameters["genomic_annotations_housekeeping_genes_dicts_directory"]

    return 0

def load_housekeeping_genes(parameters):

    housekeeping_genes = pd.read_csv("./data/Housekeeping_GenesHuman.csv", delimiter=";")

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

    with open(os.path.join(parameters["genomic_annotations_housekeeping_genes_dicts_directory"], "housekeeping_genes_" + parameters["resolution_hic_matrix_string"] + ".pickle"), 'wb') as handle:
        pickle.dump(dict_housekeeping_genes, handle, protocol=pickle.DEFAULT_PROTOCOL)

def load_dict_housekeeping_genes(parameters):
    with open(os.path.join(parameters["genomic_annotations_housekeeping_genes_dicts_directory"], "housekeeping_genes_" + parameters["resolution_hic_matrix_string"] + ".pickle"), 'rb') as handle:
        dict_housekeeping_genes = pickle.load(handle)

    return dict_housekeeping_genes


def combine_genomic_annotations_and_housekeeping_genes(genomic_annotations, housekeeping_genes):

    return 0


def generate_dict_target_genes(chr_list=None, parameters):
    if chr_list is not None:
        chromosomes = chr_list
    else:
        chromosomes = list(parameters["genomic_annotations_CTCF"].chroms().keys())
        
    dict_values = []
    
    for chromosome in chromosomes:
        chr_len = parameters["genomic_annotations_CTCF"].chroms()[chromosome]
        gbins = [x for x in range(0, chr_len, parameters["scaling_factor"])]
        
        annot_CTCF = [np.sum(parameters["genomic_annotations_CTCF"].values(chromosome, gbin, (gbin+(parameters["scaling_factor"]-1)))) for gbin in gbins[:-1]]
        annot_RAD21 = [np.sum(parameters["genomic_annotations_RAD21"].values(chromosome, gbin, (gbin+(parameters["scaling_factor"]-1)))) for gbin in gbins[:-1]]
        annot_SMC3 = [np.sum(parameters["genomic_annotations_SMC3"].values(chromosome, gbin, (gbin+(parameters["scaling_factor"]-1)))) for gbin in gbins[:-1]]

        positions = [(x+1) for x in range(int(np.floor(chr_len/parameters["scaling_factor"])))]
        values = [(annot_CTCF[i], annot_RAD21[i], annot_SMC3[i]) for i in range(len(annot_CTCF))]
        dict_values.append(dict(zip(positions, values)))
    
    dict_target_genes = dict(zip(chromosomes, dict_values))
    return dict_target_genes


def get_annotation_matrix(dictionary):
    dictionary_annotation_matrices = dict.fromkeys(dictionary.keys(), None)
    for chromosome in dictionary.keys():
        annotation_matrix = np.zeros((len(list(dictionary[chromosome].values())), 4))
        annotation_matrix[[(x-1) for x in list(dictionary[chromosome].keys())], :3] = list(dictionary[chromosome].values())
        annotation_matrix[[(x-1) for x in list(dictionary[chromosome].keys())], 3] = np.nan # INSERT values of housekeeping genes instead of nan
        dictionary_annotation_matrices[chromosome] = annotation_matrix
    return dictionary_annotation_matrices


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_parameters_json", help=" to JSON with parameters.", type=str, required=True)
    args = parser.parse_args()
    path_parameters_json = args.path_parameters_json

    parameters = load_parameters(path_parameters_json)

    for cell_line in parameters["cell_lines"]:
        if cell_line == "GM12878" or cell_line == "IMR-90":
            #if not os.path.isfile(os.path.join(parameters["genomic_annotations_housekeeping_genes_dicts_directory"], "genomic_anotations_" + cell_line + "_" + parameters["resolution_hic_matrix_string"] + ".pickle")):
            #    generate_dict_genomic_annotations(parameters, cell_line)
            # TODO
            #PREPARATION FOR STEFFI
            print("hi")
        if not os.path.isfile(os.path.join(parameters["genomic_annotations_housekeeping_genes_dicts_directory"], "housekeeping_genes_" + parameters["resolution_hic_matrix_string"] + ".pickle")):
            housekeeping_genes = load_housekeeping_genes(parameters)
            gtf = ensembl_gtf_database(parameters)
            generate_dict_housekeeping_genes(housekeeping_genes, gtf, parameters)
