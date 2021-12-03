import sys
sys.path.insert(1, './tad_detection/')
sys.path.insert(1, './tad_detection/preprocessing/')
sys.path.insert(1, './tad_detection/model/')
sys.path.insert(1, './tad_detection/evaluation/')

import numpy as np
from utils_general import load_parameters, set_up_logger
from utils_model import load_data, split_data, torch_geometric_data_generation_dataloader, generate_metrics_plots, choose_optimal_n_clust, metrics_calculation, calculate_laplacian, normalized_adjacency, save_tad_list
from mincuttad import MinCutTAD
import pandas as pd
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time

def load_optimizer(parameters):

    if parameters["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(),  #################model.parameters()
                                     lr=parameters["learning_rate"])

    elif parameters["optimizer"] == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(),  #################model.parameters()
                                        lr=parameters["learning_rate"],
                                        weight_decay=parameters["weight_decay"])
    optimizer.zero_grad()

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, min_lr=1e-5,
                                  patience=args.lr_decay_patience, verbose=True)

    return optimizer, scheduler

def save_model(model, n_clust, parameters):

    model_dict = model.state_dict()
    torch.save(model_dict, f'{parameters["output_directory"]}/models/mincut_model_{n_clust}_{parameters["dataset_name"]}_activation_function_{parameters["activation_function"]}_learning_rate_{parameters["learning_rate"]}_n_channels_{parameters["n_channels"]}_optimizer_{parameters["optimizer"]}_type_laplacian_{parameters["type_laplacian"]}_weight_decay_{parameters["weight_decay"]}.model')

def train(model, data, optimizer, parameters, device):

    # TODO
    #Make this ready for several matrices together in data.

    n_clust = 0

    n_clust_list = []

    silhouette_score_list = [] #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
    silhouette_samples_list = [] #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_samples.html#sklearn.metrics.silhouette_samples
    homogeneity_score_list = [] #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_score.html
    completeness_score_list = [] #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.completeness_score.html
    v_measure_score_list = [] #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.v_measure_score.html
    calinski_harabasz_score_list = [] #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html#sklearn.metrics.calinski_harabasz_score
    davies_bouldin_score_list = [] #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html#sklearn.metrics.davies_bouldin_score
    time_list = []
    # TODO
    #WE KIND OF ONLY WANT TO KNOW WHETHER ALL TADs ARE IN THE RIGHT CLUSTER NOT WETHER THE REST IS IN ARBITRARY CLUSTER

    #NOTSURE WHETHER NEEDED
    criterion = nn.NLLLoss()

    # Data
    X = data.X
    edge_index = data.edge_index
    labels_true = data.y

    if parameters["graph_matrix_representation"] == "laplacian":
        edge_index = calculate_laplacian(parameters["type_laplacian"], X, edge_index)
    elif parameters["graph_matrix_representation"] == "normalized":
        edge_index = normalized_adjacency(edge_index)
    else:
        raise ValueError("A graph matrix representation has been chosen, which is not implemented.")

    while n_clust < 50:
        n_clust += 1
        n_clust_list.append(n_clust)

        if len(silhouette_score_list) > 3 and np.all(silhouette_score_list[-1:-3] < max(silhouette_score_list)):
            logger.info("Clustering with " + str(n_clust) + " clusters.")

            start_time_mincutad = time.time()
            labels = model(X, edge_index)

            '''
            out, lp_loss, entropy_loss = model(data)
            out = F.log_softmax(out, dim=-1)
            loss = F.nll_loss(out, data.y.view(-1), reduction='mean')
            if model.pooling_type == 'mlp':
                (lp_loss + loss + entropy_loss).backward()
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_ce_loss += loss.item() * data.num_graphs
            total_lp_loss += lp_loss.item() * data.num_graphs
            total_e_loss += entropy_loss.item() * data.num_graphs
            return total_ce_loss / len(loader.dataset), total_lp_loss / len(loader.dataset), total_e_loss / len(loader.dataset)
            '''

            labels = np.argmax(labels, axis=1)
            end_time_mincutad = time.time()
            time_list.append(start_time_mincutad - end_time_mincutad)

            save_model(model, n_clust, parameters)

            # TODO
            predicted_tad = 0

            silhouette_score, silhouette_samples, homogeneity_score, completeness_score, v_measure_score, calinski_harabasz_score, davies_bouldin_score = metrics_calculation(
                X, labels, labels_true)
            silhouette_score_list.append(silhouette_score)
            silhouette_samples_list.append(silhouette_samples)
            homogeneity_score_list.append(homogeneity_score)
            completeness_score_list.append(completeness_score)
            v_measure_score_list.append(v_measure_score)
            calinski_harabasz_score_list.append(calinski_harabasz_score)
            davies_bouldin_score_list.append(davies_bouldin_score)

        else:
            logger.info("A maximum silhouette score has been detected. Further clustering has been stopped. Evaluation scripts are now generated.")
            break


    score_metrics_clustering = pd.DataFrame(list(zip(n_clust_list, time_list, silhouette_score_list, silhouette_samples_list, homogeneity_score_list, completeness_score_list, davies_bouldin_score_list, calinski_harabasz_score_list, davies_bouldin_score_list)),
                   columns =["Number clusters", "Calculation time algorithm", "Silhouette score", "Silhouette scores of genomic locations (bins)", "Homogeneity score", "Completeness score", "V measure score", "Calinski Harabasz score", "Davies Bouldin score"])
    score_metrics_clustering.to_pickle(os.path.join(parameters["output_directory"], "score_metrics_clustering_on_" + parameters["dataset_name"] + "_activation_function_" + parameters["activation_function"] + "_learning_rate_" + parameters["learning_rate"] + "_n_channels_" + parameters["n_channels"] + "_optimizer_" + parameters["optimizer"] + "_type_laplacian_" + parameters["type_laplacian"] + "_weight_decay_" + parameters["weight_decay"] + ".pickle"))

    return score_metrics_clustering, predicted_tad

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_parameters_json", help=" to JSON with parameters.", type=str, required=True)
    args = parser.parse_args()
    path_parameters_json = args.path_parameters_json
    # path_parameters_json = ./tad_detection/model/

    parameters = load_parameters(path_parameters_json)
    os.makedirs(parameters["output_directory"], exist_ok=True)
    os.makedirs(os.path.join(parameters["output_directory"], "training"), exist_ok=True)
    os.makedirs(os.path.join(parameters["output_directory"], "models"), exist_ok=True)
    logger = set_up_logger('training', parameters)
    logger.debug('Start training logger.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, edge_index, y = load_data(parameters, device)
    data_train_cross_1, data_test_cross_1, data_val_cross_1, data_train_cross_2, data_test_cross_2, data_val_cross_2 = split_data(parameters, X, edge_index, y)
    dataloader_train_cross_1, dataloader_test_cross_1, dataloader_val_cross_1, dataloader_train_cross_2, dataloader_test_cross_2, dataloader_val_cross_2 = torch_geometric_data_generation_dataloader(data_train_cross_1, data_test_cross_1, data_val_cross_1, data_train_cross_2, data_test_cross_2, data_val_cross_2)

    model = MinCutTAD(parameters).to(device)
    optimizer, scheduler = load_optimizer(parameters)

    model.train()
    score_metrics_clustering, predicted_tad = train(model, data, optimizer, parameters, device)

    model.evaluate()



    generate_metrics_plots(score_metrics_clustering)

    optimal_n_clust = choose_optimal_n_clust()

    save_tad_list(parameters, predicted_tad[optimal_n_clust], "MinCutTAD")