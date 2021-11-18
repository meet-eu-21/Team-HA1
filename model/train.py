import numpy as np
from sklearn.cluster import spectral_clustering
from utils import load_parameters, set_up_logger, load_data, generate_metrics_plots, choose_optimal_n_clust, metrics_calculation, calculate_laplacian, normalized_adjacency, save_tad_list
from model import MinCutTAD
import pandas as pd
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


def train(model, data, parameters, device):

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

    silhouette_score_list_baseline = []
    silhouette_samples_list_baseline = []
    homogeneity_score_list_baseline = []
    completeness_score_list_baseline = []
    v_measure_score_list_baseline = []
    calinski_harabasz_score_list_baseline = []
    davies_bouldin_score_list_baseline = []
    time_list_baseline = []

    if parameters["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(),  #################model.parameters()
                                     lr=parameters["learning_rate"])
    elif parameters["optimizer"] == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(),  #################model.parameters()
                                        lr=parameters["learning_rate"],
                                        weight_decay=parameters["weight_decay"])

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
            labels = MinCutTAD(X, edge_index)
            labels = np.argmax(labels, axis=1)
            end_time_mincutad = time.time()
            time_list.append(start_time_mincutad - end_time_mincutad)

            silhouette_score, silhouette_samples, homogeneity_score, completeness_score, v_measure_score, calinski_harabasz_score, davies_bouldin_score = metrics_calculation(
                X, labels, labels_true)
            silhouette_score_list.append(silhouette_score)
            silhouette_samples_list.append(silhouette_samples)
            homogeneity_score_list.append(homogeneity_score)
            completeness_score_list.append(completeness_score)
            v_measure_score_list.append(v_measure_score)
            calinski_harabasz_score_list.append(calinski_harabasz_score)
            davies_bouldin_score_list.append(davies_bouldin_score)

            #Baseline experiment
            start_time_baseline = time.time()
            labels = spectral_clustering(X, n_clusters=n_clust, eigen_solver='arpack')
            end_time_baseline = time.time()
            time_list_baseline.append(start_time_baseline - end_time_baseline)
            # Check whether result really is labels


            silhouette_score, silhouette_samples, homogeneity_score, completeness_score, v_measure_score, calinski_harabasz_score, davies_bouldin_score = metrics_calculation(
                X, labels, labels_true)
            silhouette_score_list_baseline.append(silhouette_score)
            silhouette_samples_list_baseline.append(silhouette_samples)
            homogeneity_score_list_baseline.append(homogeneity_score)
            completeness_score_list_baseline.append(completeness_score)
            v_measure_score_list_baseline.append(v_measure_score)
            calinski_harabasz_score_list_baseline.append(calinski_harabasz_score)
            davies_bouldin_score_list_baseline.append(davies_bouldin_score)

        else:
            logger.info("A maximum silhouette score has been detected. Further clustering has been stopped. Evaluation scripts are now generated.")
            break


    score_metrics_clustering = pd.DataFrame(list(zip(n_clust_list, time_list, silhouette_score_list, silhouette_samples_list, homogeneity_score_list, completeness_score_list, davies_bouldin_score_list, calinski_harabasz_score_list, davies_bouldin_score_list)),
                   columns =["Number clusters", "Calculation time algorithm", "Silhouette score", "Silhouette scores of genomic locations (bins)", "Homogeneity score", "Completeness score", "V measure score", "Calinski Harabasz score", "Davies Bouldin score"])
    score_metrics_clustering.to_pickle(os.path.join(parameters["output_directory"], "score_metrics_clustering_on_" + parameters["dataset_name"] + "_activation_function_" + parameters["activation_function"] + "_learning_rate_" + parameters["learning_rate"] + "_n_channels_" + parameters["n_channels"] + "_optimizer_" + parameters["optimizer"] + "_type_laplacian_" + parameters["type_laplacian"] + "_weight_decay_" + parameters["weight_decay"] + ".pickle"))

    score_metrics_clustering_baseline = pd.DataFrame(list(zip(n_clust_list, silhouette_score_list_baseline, silhouette_samples_list_baseline, homogeneity_score_list_baseline, completeness_score_list_baseline, davies_bouldin_score_list_baseline, calinski_harabasz_score_list_baseline, davies_bouldin_score_list_baseline)),
                   columns =["Number clusters", "Calculation time algorithm", "Silhouette score", "Silhouette scores of genomic locations (bins)", "Homogeneity score", "Completeness score", "V measure score", "Calinski Harabasz score", "Davies Bouldin score"])
    score_metrics_clustering_baseline.to_pickle(os.path.join(parameters["output_directory"], "score_metrics_clustering_baseline.pickle"))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_parameters_json", help=" to JSON with parameters.", type=str, required=True)
    args = parser.parse_args()
    path_parameters_json = args.path_parameters_json

    parameters = load_parameters(path_parameters_json)
    os.makedirs(parameters["output_path"], exist_ok=True)
    os.makedirs(os.path.join(parameters["output_path"], "training"), exist_ok=True)
    set_up_logger(parameters)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = load_data(parameters, device)
    model = MinCutTAD(parameters).to(device)

    score_metrics_clustering, score_metrics_clustering_baseline, predicted_tad = train(model, data, parameters, device)

    generate_metrics_plots(score_metrics_clustering)
    generate_metrics_plots(score_metrics_clustering_baseline)

    optimal_n_clust = choose_optimal_n_clust()

    save_tad_list(parameters, predicted_tad[optimal_n_clust], "MinCutTAD")