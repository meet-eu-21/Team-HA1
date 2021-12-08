import sys
sys.path.insert(1, './tad_detection/')
sys.path.insert(1, './tad_detection/preprocessing/')
sys.path.insert(1, './tad_detection/model/')
sys.path.insert(1, './tad_detection/evaluation/')

import logging
logger = logging.getLogger('training')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples, homogeneity_score, completeness_score, v_measure_score, calinski_harabasz_score, davies_bouldin_score
from torch_geometric.utils import get_laplacian
from sklearn.decomposition import PCA
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_add
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import logging
import os
import scipy as sp
import random
import math

def _old_load_parameters(path_parameters_json):
    '''
    Function loads the parameters from the provided parameters.json file in a dictionary.

    :param path_parameters_json: path of parameters.json file
    :return parameters: dictionary with parameters set in parameters.json file.
    '''

    with open(path_parameters_json) as parameters_json:
        parameters = json.load(parameters_json)

    return parameters


def _old_set_up_logger(parameters):
    '''
    Function sets a global logger for documentation of information and errors in the execution of the chosen script.

    :param parameters: dictionary with parameters set in parameters.json file.
    '''

    logger = logging.getLogger('training')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(parameters["output_directory"], 'training', 'training.log'))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    logger.info("Training with the following parameters:")
    for parameter in parameters.keys():
        logger.info(parameter + ": " + str(parameters[parameter]))

def load_data(parameters):
    '''

    :param parameters: dictionary with parameters set in parameters.json file
    :return:
    '''

    X = np.load(os.path.join(parameters["dataset_path"], parameters["dataset_name"], parameters["dataset_name"] + "_X.npy"), allow_pickle=True)
    edge_index = np.load(os.path.join(parameters["dataset_path"], parameters["dataset_name"], parameters["dataset_name"] + "_edge_index.npy"), allow_pickle=True)
    y = np.load(os.path.join(parameters["dataset_path"], parameters["dataset_name"], parameters["dataset_name"] + "_y.npy"), allow_pickle=True)

    if X.shape[1] != edge_index.shape[1]:
        raise ValueError('The shape of X and the edge_index does not fit together.')
    if X.shape[1] != y.shape[1]:
        raise ValueError('The shape of X and y does not fit together.')

    return X, edge_index, y

def split_data(parameters, X, edge_index, y):

    if parameters["proportion_val_set"] + parameters["proportion_test_set"] + parameters["proportion_train_set"] == 1:
        if X.shape[0] == 2:
            logger.info("Two cell lines are inputted. A cross-validation on both datasets will be performed. One cell line is used for training and the other one for testing and validation.")

            data_train_list_cross_1 = []
            data_train_list_cross_2 = []
            data_test_list_cross_1 = []
            data_test_list_cross_2 = []
            data_val_list_cross_1 = []
            data_val_list_cross_2 = []

            for chromosome in range(X[0].shape[0]):
                data_train_list_cross_1.append(Data(x=torch.from_numpy(X[0][chromosome]), edge_index=torch.from_numpy(edge_index[0][chromosome]), y=torch.from_numpy(y[0][chromosome])))
            for chromosome in range(X[1].shape[0]):
                data_train_list_cross_2.append(Data(x=torch.from_numpy(X[1][chromosome]), edge_index=torch.from_numpy(edge_index[1][chromosome]), y=torch.from_numpy(y[1][chromosome])))

            test_chromosomes = random.sample(list(range(0, X[0].shape[0])), math.floor(np.round(X[0].shape[0]/2)))
            val_chromosomes = list(set(range(0, X[0].shape[0])) - set(test_chromosomes))

            for chromosome in test_chromosomes:
                data_test_list_cross_1.append(Data(x=torch.from_numpy(X[1][chromosome]), edge_index=torch.from_numpy(edge_index[1][chromosome]), y=torch.from_numpy(y[1][chromosome]))) #[test_chromosomes,:]
                data_test_list_cross_2.append(Data(x=torch.from_numpy(X[0][chromosome]), edge_index=torch.from_numpy(edge_index[0][chromosome]), y=torch.from_numpy(y[0][chromosome]))) #[test_chromosomes,:]

            for chromosome in val_chromosomes:
                data_val_list_cross_1.append(Data(x=torch.from_numpy(X[1][chromosome]), edge_index=torch.from_numpy(edge_index[1][chromosome]), y=torch.from_numpy(y[1][chromosome]))) #[val_chromosomes,:]
                data_val_list_cross_2.append(Data(x=torch.from_numpy(X[0][chromosome]), edge_index=torch.from_numpy(edge_index[0][chromosome]), y=torch.from_numpy(y[0][chromosome]))) #[val_chromosomes,:]

            return data_train_list_cross_1, data_test_list_cross_1, data_val_list_cross_1, data_train_list_cross_2, data_test_list_cross_2, data_val_list_cross_2

        else:

            val_chromosomes = random.sample(list(range(0, X[0].shape[0])), math.floor(X[0].shape[0] * parameters["proportion_val_set"]))
            test_chromosomes = random.sample(list(set(range(0, X[0].shape[0])) - set(val_chromosomes)), math.floor(X[0].shape[0] * parameters["proportion_test_set"]))
            train_chromosomes = list(set(range(0, X[0].shape[0])) - set(val_chromosomes) - set(test_chromosomes))

            data_train_list = []
            data_test_list = []
            data_val_list = []

            for chromosome in train_chromosomes:
                data_train_list.append(Data(x=torch.from_numpy(X[0][chromosome]), edge_index=torch.from_numpy(edge_index[0][chromosome]), y=torch.from_numpy(y[0][chromosome])))
            for chromosome in test_chromosomes:
                data_test_list.append(Data(x=torch.from_numpy(X[0][chromosome]), edge_index=torch.from_numpy(edge_index[0][chromosome]), y=torch.from_numpy(y[0][chromosome])))
            for chromosome in val_chromosomes:
                data_val_list.append(Data(x=torch.from_numpy(X[0][chromosome]), edge_index=torch.from_numpy(edge_index[0][chromosome]), y=torch.from_numpy(y[0][chromosome])))

            # x: .astype("float32")
            # edge_index: .type(torch.LongTensor)
            # y: .type(torch.LongTensor)

            return data_train_list, data_test_list, data_val_list, 0, 0, 0

    else:
        raise NotImplementedError("The validation ,training and test data sizes do not sum up to 1. Training aborted.")


def torch_geometric_data_generation_dataloader(data_train_cross_1, data_test_cross_1, data_val_cross_1, data_train_cross_2, data_test_cross_2, data_val_cross_2):

    dataloader_train_cross_1 = DataLoader(data_train_cross_1, batch_size=1)
    dataloader_test_cross_1 = DataLoader(data_test_cross_1, batch_size=1)
    dataloader_val_cross_1 = DataLoader(data_val_cross_1, batch_size=1)

    if all([data_train_cross_2, data_train_cross_2, data_train_cross_2]) == 0:

        return dataloader_train_cross_1, dataloader_test_cross_1, dataloader_val_cross_1, 0, 0, 0

    else:

        dataloader_train_cross_2 = DataLoader(data_train_cross_2, batch_size=1)
        dataloader_test_cross_2 = DataLoader(data_test_cross_2, batch_size=1)
        dataloader_val_cross_2 = DataLoader(data_val_cross_2, batch_size=1)

        return dataloader_train_cross_1, dataloader_test_cross_1, dataloader_val_cross_1, dataloader_train_cross_2, dataloader_test_cross_2, dataloader_val_cross_2


def load_optimizer(model, parameters_user):

    if parameters_user["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=parameters_user["learning_rate"])

    elif parameters_user["optimizer"] == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(),
                                        lr=parameters_user["learning_rate"],
                                        weight_decay=parameters_user["weight_decay"])
    optimizer.zero_grad()

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, min_lr=1e-5,
                                  patience=parameters_user["learning_rate_decay_patience"], verbose=True)

    return optimizer, scheduler

def save_model(model, epoch_num, n_clust, parameters):

    model_dict = model.state_dict()
    torch.save(model_dict, f'{parameters["output_directory"]}/models/mincut_model_{n_clust}_{parameters["dataset_name"]}_epoch_{epoch_num}_activation_function_{parameters["activation_function"]}_learning_rate_{parameters["learning_rate"]}_n_channels_{parameters["n_channels"]}_optimizer_{parameters["optimizer"]}_type_laplacian_{parameters["type_laplacian"]}_weight_decay_{parameters["weight_decay"]}.model')

def load_model(n_clust_test_optimum, epoch_num_optimum, parameters):

    model = torch.load(f'{parameters["output_directory"]}/models/mincut_model_{n_clust_test_optimum}_{parameters["dataset_name"]}_epoch_{epoch_num_optimum}_activation_function_{parameters["activation_function"]}_learning_rate_{parameters["learning_rate"]}_n_channels_{parameters["n_channels"]}_optimizer_{parameters["optimizer"]}_type_laplacian_{parameters["type_laplacian"]}_weight_decay_{parameters["weight_decay"]}.model')

    return model

def generate_metrics_plots(score_metrics_clustering, output_directory):
    '''

    :param score_metrics_clustering:
    :param output_directory:
    :return:
    '''

    for metric in list(score_metrics_clustering.columns)[1:]:
        plt.plot("Number clusters", metric, data=score_metrics_clustering)
        plt.xlabel("Number clusters")
        plt.ylabel(metric)
        #plt.show()
        plt.plt.savefig(os.path.join(output_directory, "training", metric + "_vs_n_clust.png"))

    for metric in list(score_metrics_clustering.columns)[1:]:
        plt.plot("Number clusters", metric, data=score_metrics_clustering)
    plt.xlabel("Number clusters")
    plt.ylabel("Values of metric")
    plt.legend()
    #plt.show()
    plt.plt.savefig(os.path.join(output_directory, "training", "All_metrics_vs_n_clust.png"))

def choose_optimal_n_clust(silhouette_score_list):
    '''

    :param silhouette_score_list:
    :return:
    '''

    optimal_n_clust = np.where(np.array(silhouette_score_list) == max(silhouette_score_list))[0][0]

    return optimal_n_clust

def metrics_calculation(X, labels, labels_true):
    '''

    :param X:
    :param labels:
    :param labels_true:
    :return:
    '''

    # labels-array-like of shape (n_samples,) - Predicted labels for each sample.

    silhouette_score_calc = silhouette_score(X, labels)  # The Silhouette Coefficient is calculated using the mean intra-cluster distance (a) and the mean nearest-cluster distance (b) for each sample. The Silhouette Coefficient for a sample is (b - a) / max(a, b). To clarify, b is the distance between a sample and the nearest cluster that the sample is not a part of. Note that Silhouette Coefficient is only defined if number of labels is 2 <= n_labels <= n_samples - 1. The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters. Negative values generally indicate that a sample has been assigned to the wrong cluster, as a different cluster is more similar.
    # Xarray-like of shape (n_samples_a, n_samples_a) if metric == “precomputed” or (n_samples_a, n_features) otherwise - An array of pairwise distances between samples, or a feature array.
    silhouette_samples_calc = silhouette_samples(X, labels)  # The Silhouette Coefficient is a measure of how well samples are clustered with samples that are similar to themselves. Clustering models with a high Silhouette Coefficient are said to be dense, where samples in the same cluster are similar to each other, and well separated, where samples in different clusters are not very similar to each other. - The Silhouette Coefficient is calculated using the mean intra-cluster distance (a) and the mean nearest-cluster distance (b) for each sample. The Silhouette Coefficient for a sample is (b - a) / max(a, b). Note that Silhouette Coefficient is only defined if number of labels is 2 <= n_labels <= n_samples - 1. - This function returns the Silhouette Coefficient for each sample. - The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters.
    # Xarray-like of shape (n_samples_a, n_samples_a) if metric == “precomputed” or (n_samples_a, n_features) otherwise - An array of pairwise distances between samples, or a feature array.
    homogeneity_score_calc = homogeneity_score(labels, labels_true)  # A clustering result satisfies homogeneity if all of its clusters contain only data points which are members of a single class. This metric is independent of the absolute values of the labels: a permutation of the class or cluster label values won’t change the score value in any way.
    completeness_score_calc = completeness_score(labels, labels_true)  # A clustering result satisfies completeness if all the data points that are members of a given class are elements of the same cluster. - This metric is independent of the absolute values of the labels: a permutation of the class or cluster label values won’t change the score value in any way.
    v_measure_score_calc = v_measure_score(labels, labels_true)  # The V-measure is the harmonic mean between homogeneity and completeness: v = (1 + beta) * homogeneity * completeness / (beta * homogeneity + completeness)
    calinski_harabasz_score_calc = calinski_harabasz_score(X, labels)  # The score is defined as ratio between the within-cluster dispersion and the between-cluster dispersion.
    # X-array-like of shape (n_samples, n_features) - A list of n_features-dimensional data points. Each row corresponds to a single data point.
    davies_bouldin_score_calc = davies_bouldin_score(X, labels)  # The score is defined as the average similarity measure of each cluster with its most similar cluster, where similarity is the ratio of within-cluster distances to between-cluster distances. Thus, clusters which are farther apart and less dispersed will result in a better score. The minimum score is zero, with lower values indicating better clustering.
    # X-array-like of shape (n_samples, n_features) - A list of n_features-dimensional data points. Each row corresponds to a single data point.

    return silhouette_score_calc, silhouette_samples_calc, homogeneity_score_calc, completeness_score_calc, v_measure_score_calc, calinski_harabasz_score_calc, davies_bouldin_score_calc

def calculate_laplacian(type_laplacian, edge_index, X):
    '''

    :param type_laplacian:
    :param edge_index:
    :param X:
    :return:
    '''

    if type_laplacian == "unweighted_laplacian":
        get_laplacian(edge_index = edge_index, normalization = "sym")
    elif type_laplacian == "weighted_laplacian":
        pca = PCA(n_components=1)
        edge_weight = pca.fit_transform(X)
        logger.info("Explained variance ratio: " + str(pca.explained_variance_ratio_))
        logger.info("Explained singular values: " + str(pca.singular_values_))
        edge_index = get_laplacian(edge_index = edge_index, edge_weight = edge_weight, normalization = "sym")

    '''
    Alternatives:
    from scipy.sparse.csgraph import laplacian

    laplacian(G, normed=True) #Edge weight is not taken into account #Normalization done in cresswell #Doc: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.laplacian.html

    networkx.linalg.laplacianmatrix.normalized_laplacian_matrix
    normalized_laplacian_matrix(G, nodelist=None, weight='weight')
    #The edge data key used to compute each value in the matrix. If None, then each edge has weight 1.

    networkx.linalg.laplacianmatrix.laplacian_matrix
    laplacian_matrix(G, nodelist=None, weight='weight')
    '''

    return edge_index

'''
def get_laplacian(edge_index, edge_weight: Optional[torch.Tensor] = None,
                  normalization: Optional[str] = None,
                  dtype: Optional[int] = None,
                  num_nodes: Optional[int] = None):
    r""" Computes the graph Laplacian of the graph given by :obj:`edge_index`
    and optional :obj:`edge_weight`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_weight (Tensor, optional): One-dimensional edge weights.
            (default: :obj:`None`)
        normalization (str, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`None`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`
        dtype (torch.dtype, optional): The desired data type of returned tensor
            in case :obj:`edge_weight=None`. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    """

    if normalization is not None:
        assert normalization in ['sym', 'rw']  # 'Invalid normalization'

    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=dtype,
                                 device=edge_index.device)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)

    if normalization is None:
        # L = D - A.
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        edge_weight = torch.cat([-edge_weight, deg], dim=0)
    elif normalization == 'sym':
        # Compute A_norm = -D^{-1/2} A D^{-1/2}.
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        # L = I - A_norm.
        edge_index, tmp = add_self_loops(edge_index, -edge_weight,
                                         fill_value=1., num_nodes=num_nodes)
        assert tmp is not None
        edge_weight = tmp
    else:
        # Compute A_norm = -D^{-1} A.
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight

        # L = I - A_norm.
        edge_index, tmp = add_self_loops(edge_index, -edge_weight,
                                         fill_value=1., num_nodes=num_nodes)
        assert tmp is not None
        edge_weight = tmp

    return edge_index, edge_weight
'''

def normalized_adjacency(edge_index):
    '''

    :param edge_index:
    :return:
    '''
    #INSPIRATION: https://github.com/danielegrattarola/spektral/blob/e99d5955a80eeae3c4605d8479f53aaa0ef5dbf2/spektral/utils/convolution.py#L25
    degrees = np.power(np.array(edge_index.sum(1)), -0.5).ravel()
    degrees[np.isinf(degrees)] = 0.0
    if sp.issparse(edge_index):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D.dot(edge_index).dot(D)

'''
#ORIGINAL - SPECTRAL
def normalized_adjacency(A, symmetric=True):
    r"""
    Normalizes the given adjacency matrix using the degree matrix as either
    \(\D^{-1}\A\) or \(\D^{-1/2}\A\D^{-1/2}\) (symmetric normalization).
    :param A: rank 2 array or sparse matrix;
    :param symmetric: boolean, compute symmetric normalization;
    :return: the normalized adjacency matrix.
    """
    if symmetric:
        normalized_D = degree_power(A, -0.5)
        return normalized_D.dot(A).dot(normalized_D)
    else:
        normalized_D = degree_power(A, -1.0)
        return normalized_D.dot(A)
'''

def save_tad_list(parameters, tad_list, tool):
    '''

    :param parameters: dictionary with parameters set in parameters.json file
    :param tad_list:
    :param tool:
    :return:
    '''

    np.save(os.path.join(parameters["output_directory"], "training", tool + ".npy"), tad_list)

def calculation_graph_matrix_representation(parameters, X, edge_index):

    if parameters["graph_matrix_representation"] == "laplacian":
        edge_index = calculate_laplacian(parameters["type_laplacian"], X, edge_index)
    elif parameters["graph_matrix_representation"] == "normalized":
        edge_index = normalized_adjacency(edge_index)
    else:
        raise ValueError("A graph matrix representation has been chosen, which is not implemented.")

    return edge_index