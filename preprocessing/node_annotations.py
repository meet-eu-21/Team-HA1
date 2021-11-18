import pandas as pd
import numpy as np

def load_genomic_annotations(parameters):
    return 0


def load_housekeeping_genes(parameters):

    housekeeping_genes = pd.read_csv("./data/Housekeeping_GenesHuman.csv", delimiter=";")

    housekeeping_genes = housekeeping_genes["Gene.name"]

    return 0

def assign_housekeeping_genes_to_chromosome_bins(neighborhood_radius):
    '''

    :param neighborhood_radius:
    :return:
    '''

    return 0

def combine_genomic_annotations_and_housekeeping_genes(genomic_annotations, housekeeping_genes):

    return 0


