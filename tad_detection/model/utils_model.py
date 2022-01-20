import sys
sys.path.insert(1, './tad_detection/')
sys.path.insert(1, './tad_detection/preprocessing/')
sys.path.insert(1, './tad_detection/model/')
sys.path.insert(1, './tad_detection/evaluation/')

import logging
logger = logging.getLogger('training')

import numpy as np
from torch_geometric.utils import get_laplacian
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import random
import math
import neptune.new as neptune
import pickle
from model.mincuttad import MinCutTAD

def set_up_neptune(parameters):
    '''
    Function sets up neptune for hyperparameter and training progress logging.

    :param parameters: dictionary with parameters set in parameters.json file
    :return run: neptune logger
    '''

    run = neptune.init(
        project="MinCutTAD/TAD",
        tags="Debugging",
        # api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmZGVkZDY5ZS03Yzg5LTQ1NmEtYWViYi1kZTgzMmJiNjViY2YifQ==",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyNDZjOTY2NS1mMjc5LTQxZjAtYjJiMC0wNTNhZDI1MmM5ZDcifQ==",
        source_files=["*.py"],
    )

    run["parameters"] = parameters

    return run

def load_data(parameters):
    '''
    Function loads X (node_features), edge_index, edge_attr, y (One-hot encoded TADs called by Arrowhead [Ground truth]) from the output directory given in parameters.

    :param parameters: dictionary with parameters set in parameters.json file
    :return X: node (genomic bins in adjacency graph) features of graph (e.g. prevalence of CTCF or one-hot encoded housekeeping genes presence)
    :return max_num_nodes: maximal number of nodes in dataset
    :return edge_index: representation of adjacency matrix of graph, gives nodes and edges between nodes
    :return edge_attr: representation of adjacency matrix of graph, gives attributes (e.g. weight) of edges in edge_index
    :return y: TAD or No-TAD labels of nodes in graph (genomic bins in adjacency graph)
    :return source_information: source informations of chromosomes and cell lines
    '''

    X = np.load(os.path.join(parameters["dataset_path"], parameters["dataset_name"], parameters["dataset_name"] + "_X.npy"), allow_pickle=True)
    edge_index = np.load(os.path.join(parameters["dataset_path"], parameters["dataset_name"], parameters["dataset_name"] + "_edge_index.npy"), allow_pickle=True)
    edge_attr = np.load(os.path.join(parameters["dataset_path"], parameters["dataset_name"], parameters["dataset_name"] + "_edge_attr.npy"), allow_pickle=True)
    y = np.load(os.path.join(parameters["dataset_path"], parameters["dataset_name"], parameters["dataset_name"] + "_y.npy"), allow_pickle=True)
    source_information = np.load(os.path.join(parameters["dataset_path"], parameters["dataset_name"], parameters["dataset_name"] + "_source_information.npy"), allow_pickle=True)

    if X.shape[1] != edge_index.shape[1]:
        raise ValueError('The shape of X and the edge_index does not fit together.')
    if X.shape[1] != y.shape[1]:
        raise ValueError('The shape of X and y does not fit together.')

    max_num_nodes = 0
    for X_cell_line in X:
        for node_attributes in X_cell_line:
            max_num_nodes = max(max_num_nodes, node_attributes.shape[0])

    return X, max_num_nodes, edge_index, edge_attr, y, source_information

def split_data(parameters, X, edge_index, edge_attr, y, source_information):
    '''
    Function splits X, edge_index, edge_attr and y in a training, test and validation dataset (Can also be used for cross-validation.).

    :param parameters: dictionary with parameters set in parameters.json file
    :param X: node (genomic bins in adjacency graph) features of graph (e.g. prevalence of CTCF or one-hot encoded housekeeping genes presence)
    :param edge_index: representation of adjacency matrix of graph, gives nodes and edges between nodes
    :param edge_attr: representation of adjacency matrix of graph, gives attributes (e.g. weight) of edges in edge_index
    :param y: TAD or No-TAD labels of nodes in graph (genomic bins in adjacency graph)
    :param source_information: source informations of chromosomes and cell lines
    :return data_train_list_cross_1: training dataset (1 for cross-validation)
    :return data_test_list_cross_1: test dataset (1 for cross-validation)
    :return data_val_list_cross_1: validation dataset (1 for cross-validation)
    :return data_train_list_cross_2: training dataset 2 for cross-validation
    :return data_test_list_cross_2: test dataset 2 for cross-validation
    :return data_val_list_cross_2: validation dataset 2 for cross-validation
    '''

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
                data_train_list_cross_1.append(Data(x=torch.from_numpy(X[0][chromosome]).float(), edge_index=torch.from_numpy(np.array([edge_index[0][chromosome][0], edge_index[0][chromosome][1]], dtype="int64")), edge_attr=torch.from_numpy(edge_attr[0][chromosome]), y=torch.from_numpy(y[0][chromosome]), source_information=source_information[0][chromosome]))
            for chromosome in range(X[1].shape[0]):
                data_train_list_cross_2.append(Data(x=torch.from_numpy(X[1][chromosome]).float(), edge_index=torch.from_numpy(np.array([edge_index[1][chromosome][0], edge_index[1][chromosome][1]], dtype="int64")), edge_attr=torch.from_numpy(edge_attr[1][chromosome]), y=torch.from_numpy(y[1][chromosome]), source_information=source_information[1][chromosome]))

            test_chromosomes = random.sample(list(range(0, X[0].shape[0])), math.floor(np.round(X[0].shape[0]/2)))
            val_chromosomes = list(set(range(0, X[0].shape[0])) - set(test_chromosomes))

            for chromosome in test_chromosomes:
                data_test_list_cross_1.append(Data(x=torch.from_numpy(X[1][chromosome]).float(), edge_index=torch.from_numpy(np.array([edge_index[1][chromosome][0], edge_index[1][chromosome][1]], dtype="int64")), edge_attr=torch.from_numpy(edge_attr[1][chromosome]), y=torch.from_numpy(y[1][chromosome]), source_information=source_information[1][chromosome]))
                data_test_list_cross_2.append(Data(x=torch.from_numpy(X[0][chromosome]).float(), edge_index=torch.from_numpy(np.array([edge_index[0][chromosome][0], edge_index[0][chromosome][1]], dtype="int64")), edge_attr=torch.from_numpy(edge_attr[0][chromosome]), y=torch.from_numpy(y[0][chromosome]), source_information=source_information[0][chromosome]))

            for chromosome in val_chromosomes:
                data_val_list_cross_1.append(Data(x=torch.from_numpy(X[1][chromosome]).float(), edge_index=torch.from_numpy(np.array([edge_index[1][chromosome][0], edge_index[1][chromosome][1]], dtype="int64")), edge_attr=torch.from_numpy(edge_attr[1][chromosome]), y=torch.from_numpy(y[1][chromosome]), source_information=source_information[1][chromosome]))
                data_val_list_cross_2.append(Data(x=torch.from_numpy(X[0][chromosome]).float(), edge_index=torch.from_numpy(np.array([edge_index[0][chromosome][0], edge_index[0][chromosome][1]], dtype="int64")), edge_attr=torch.from_numpy(edge_attr[0][chromosome]), y=torch.from_numpy(y[0][chromosome]), source_information=source_information[0][chromosome]))

            return data_train_list_cross_1, data_test_list_cross_1, data_val_list_cross_1, data_train_list_cross_2, data_test_list_cross_2, data_val_list_cross_2

        else:

            val_chromosomes = random.sample(list(range(0, X[0].shape[0])), math.floor(X[0].shape[0] * parameters["proportion_val_set"]))
            test_chromosomes = random.sample(list(set(range(0, X[0].shape[0])) - set(val_chromosomes)), math.floor(X[0].shape[0] * parameters["proportion_test_set"]))
            train_chromosomes = list(set(range(0, X[0].shape[0])) - set(val_chromosomes) - set(test_chromosomes))

            data_train_list = []
            data_test_list = []
            data_val_list = []

            for chromosome in train_chromosomes:
                data_train_list.append(Data(x=torch.from_numpy(X[0][chromosome]).float(), edge_index=torch.from_numpy(np.array([edge_index[0][chromosome][0], edge_index[0][chromosome][1]], dtype="int64")), edge_attr=torch.from_numpy(edge_attr[0][chromosome]).long(), y=torch.from_numpy(y[0][chromosome]).int(), source_information=source_information[0][chromosome]))
            for chromosome in test_chromosomes:
                data_test_list.append(Data(x=torch.from_numpy(X[0][chromosome]).float(), edge_index=torch.from_numpy(np.array([edge_index[0][chromosome][0], edge_index[0][chromosome][1]], dtype="int64")), edge_attr=torch.from_numpy(edge_attr[0][chromosome]).long(), y=torch.from_numpy(y[0][chromosome]).int(), source_information=source_information[0][chromosome]))
            for chromosome in val_chromosomes:
                data_val_list.append(Data(x=torch.from_numpy(X[0][chromosome]).float(), edge_index=torch.from_numpy(np.array([edge_index[0][chromosome][0], edge_index[0][chromosome][1]], dtype="int64")), edge_attr=torch.from_numpy(edge_attr[0][chromosome]).long(), y=torch.from_numpy(y[0][chromosome]).int(), source_information=source_information[0][chromosome]))

            return data_train_list, data_test_list, data_val_list, 0, 0, 0

    else:
        raise NotImplementedError("The validation, training and test data sizes do not sum up to 1. Training aborted.")


def torch_geometric_data_generation_dataloader(data_train_cross_1, data_test_cross_1, data_val_cross_1, data_train_cross_2, data_test_cross_2, data_val_cross_2):
    '''
    Function generates dataloaders from training, test and validation datasets.

    :param data_train_cross_1: training dataset (1 for cross-validation)
    :param data_test_cross_1: test dataset (1 for cross-validation)
    :param data_val_cross_1: validation dataset (1 for cross-validation)
    :param data_train_cross_2: training dataset 2 for cross-validation
    :param data_test_cross_2: test dataset 2 for cross-validation
    :param data_val_cross_2: validation dataset 2 for cross-validation
    :return dataloader_train_cross_1: dataloader training (1 for cross-validation)
    :return dataloader_test_cross_1: dataloader training (1 for cross-validation)
    :return dataloader_val_cross_1: dataloader testing (1 for cross-validation)
    :return dataloader_train_cross_2: dataloader training 2 for cross-validation
    :return dataloader_test_cross_2: dataloader testing 2 for cross-validation
    :return dataloader_val_cross_2: dataloader validation 2 for cross-validation
    '''

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
    '''
    Function loads an optimizer and scheduler for the model.

    :param model: Untrained PyTorch model
    :param parameters_user: dictionary with parameters set in parameters.json file
    :return optimizer: PyTorch optimizer
    :return scheduler: PyTorch scheduler
    '''

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
    '''
    Function saves a trained model for a specific n_clust and epoch_num.

    :param model: PyTorch model with trained weights
    :param epoch_num: epoch number
    :param n_clust: number of clusters for MinCutTAD prediction
    :param parameters: dictionary with parameters set in parameters.json file
    '''

    model_dict = model.state_dict()

    if n_clust:
        torch.save(model_dict, f'{os.path.join(parameters["output_directory"], parameters["dataset_name"], "models")}/mincut_model_{n_clust}_{parameters["dataset_name"]}_epoch_{epoch_num}_activation_function_{parameters["activation_function"]}_learning_rate_{parameters["learning_rate"]}_n_channels_{parameters["n_channels"]}_optimizer_{parameters["optimizer"]}_type_laplacian_{parameters["type_laplacian"]}_weight_decay_{parameters["weight_decay"]}.model')
    else:
        torch.save(model_dict, f'{os.path.join(parameters["output_directory"], parameters["dataset_name"], "models")}/mincut_model_{parameters["dataset_name"]}_epoch_{epoch_num}_activation_function_{parameters["activation_function"]}_learning_rate_{parameters["learning_rate"]}_n_channels_{parameters["n_channels"]}_optimizer_{parameters["optimizer"]}_type_laplacian_{parameters["type_laplacian"]}_weight_decay_{parameters["weight_decay"]}.model')

def load_model(parameters, epoch_num, n_clust=None):
    '''
    Function loads trained model for a specific n_clust and epoch_num.

    :param n_clust: number of clusters for MinCutTAD prediction
    :param epoch_num: epoch number
    :param parameters: dictionary with parameters set in parameters.json file
    :return model: PyTorch model with trained weights
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if n_clust:
        n_clust = int(n_clust)
        model_for_test = MinCutTAD(parameters, n_clust).to(device)
        model_for_test.load_state_dict(torch.load(f'{os.path.join(parameters["output_directory"], parameters["dataset_name"], "models")}/mincut_model_{n_clust}_{parameters["dataset_name"]}_epoch_{epoch_num}_activation_function_{parameters["activation_function"]}_learning_rate_{parameters["learning_rate"]}_n_channels_{parameters["n_channels"]}_optimizer_{parameters["optimizer"]}_type_laplacian_{parameters["type_laplacian"]}_weight_decay_{parameters["weight_decay"]}.model'))
    else:
        model_for_test = MinCutTAD(parameters).to(device)
        model_for_test.load_state_dict(torch.load(f'{os.path.join(parameters["output_directory"], parameters["dataset_name"], "models")}/mincut_model_{parameters["dataset_name"]}_epoch_{epoch_num[0]}_activation_function_{parameters["activation_function"]}_learning_rate_{parameters["learning_rate"]}_n_channels_{parameters["n_channels"]}_optimizer_{parameters["optimizer"]}_type_laplacian_{parameters["type_laplacian"]}_weight_decay_{parameters["weight_decay"]}.model'))

    model_for_test.eval()

    return model_for_test

def calculate_laplacian(edge_index):
    '''
    Function calculates laplacian of adjacency matrix, which is represented by the edge_index and edge_attr. In the "weighted laplacian" the edge_attr is taken into account.

    :param edge_index: edge index representation of adjacency_matrix, reports the nodes and the edges between the nodes present in a graph
    :return edge_index: laplacian-modified version of edge index representation of adjacency_matrix, reports the nodes and the edges between the nodes present in a graph
    '''

    edge_index = get_laplacian(edge_index = edge_index, normalization = "sym")

    return edge_index

def normalized_adjacency(edge_index):
    '''
    Function calculates a normalized version of the adjacency matrix, which is represented by the edge_index.

    :param edge_index: edge index representation of adjacency_matrix, reports the nodes and the edges between the nodes present in a graph
    :return edge_index: normalized version of edge index representation of adjacency_matrix, reports the nodes and the edges between the nodes present in a graph
    '''
    #INSPIRATION: https://github.com/danielegrattarola/spektral/blob/e99d5955a80eeae3c4605d8479f53aaa0ef5dbf2/spektral/utils/convolution.py#L25
    degrees = np.power(np.array(edge_index.sum(1)), -0.5).ravel()
    degrees[np.isinf(degrees)] = 0.0
    #if sp.issparse(edge_index):
    #    D = sp.diags(degrees)
    #else:
    D = np.diag(degrees)
    return D.dot(edge_index).dot(D)

def save_tad_list(parameters, tad_list_source_information_dict, dataloader, tool, extension = None):
    '''

    :param parameters: dictionary with parameters set in parameters.json file
    :param tad_list_source_information_dict:
    :param tool:
    :return:
    '''

    cell_line = list(dataloader)[0].source_information[0].split("-")[0]
    tool = tool + "_" + list(dataloader)[0].source_information[0].split("-")[0]

    source_informations = []
    for data in list(dataloader):
        source_informations.append(data.source_information[0].split("-")[1])
    source_informations = np.array(source_informations)

    tad_list = []
    for tads in tad_list_source_information_dict[cell_line].keys():
        tad_list.append(tad_list_source_information_dict[cell_line][tads])
    tad_list = np.array(tad_list)

    if extension:
        np.save(os.path.join(parameters["output_directory"], parameters["dataset_name"], "predicted_tads_" + extension + "_" + tool + ".npy"), tad_list)
        np.save(os.path.join(parameters["output_directory"], parameters["dataset_name"], "predicted_tads_source_information_" + extension + "_" + tool + ".npy"), source_informations)
    else:
        np.save(os.path.join(parameters["output_directory"], parameters["dataset_name"], "predicted_tads_" + tool + ".npy"), tad_list)
        np.save(os.path.join(parameters["output_directory"], parameters["dataset_name"], "predicted_tads_source_information_" + tool + ".npy"), source_informations)

    with open(os.path.join(parameters["output_directory"], parameters["dataset_name"], "predicted_tads_" + tool + ".pickle"), 'wb') as handle:
        pickle.dump(tad_list_source_information_dict, handle, protocol=pickle.DEFAULT_PROTOCOL)

def calculation_graph_matrix_representation(parameters, edge_index):
    '''
    Function is a wrapper for the calculation of laplacian or normalized version of of adjacency matrix.
    '''

    if parameters["graph_matrix_representation"] == "laplacian":
        edge_index = calculate_laplacian(edge_index)
        return edge_index
    elif parameters["graph_matrix_representation"] == "normalized":
        edge_index = normalized_adjacency(edge_index)
        return edge_index, None
    else:
        raise ValueError("A graph matrix representation has been chosen, which is not implemented.")

def determine_tad_regions(predicted_tads_dict):
    '''
    Function determines TAD regions out of a dict of classified TAD bins.

    :param predicted_tads_dict: dict with TAD bins for each chromosome
    :return predicted_tads_dict: dict with TAD regions for each chromosome
    '''

    for cell_line in predicted_tads_dict.keys():
        for chromosome in predicted_tads_dict[cell_line].keys():
            tad_regions = predicted_tads_dict[cell_line][chromosome]

            array_length = len(tad_regions)
            length = 1
            predicted_tads = []

            if (array_length == 0):
                return predicted_tads

            for i in range(1, array_length + 1):
                if (i == array_length or tad_regions[i] - tad_regions[i - 1] != 1):
                    if (length == 1):
                        predicted_tads.append([(tad_regions[i - length])])
                    else:
                        temp = np.arange((tad_regions[i - length]), (tad_regions[i - 1]) + 1)
                        predicted_tads.append(temp.tolist())
                    length = 1
                else:
                    length += 1

            predicted_tads_dict[cell_line][chromosome] = predicted_tads

    return predicted_tads_dict
