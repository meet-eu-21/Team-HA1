import sys
sys.path.insert(1, './tad_detection/')
sys.path.insert(1, './tad_detection/preprocessing/')
sys.path.insert(1, './tad_detection/model/')
sys.path.insert(1, './tad_detection/evaluation/')

import seaborn as sns
import numpy as np
import pandas as pd
import os
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score, v_measure_score, calinski_harabasz_score, davies_bouldin_score, accuracy_score, precision_score, roc_auc_score, f1_score
from torch_geometric.utils import to_dense_adj
from model.gnn_explainer import GNNExplainer
from utils_general import load_parameters, dump_parameters, set_up_logger
import argparse
from model.utils_model import calculation_graph_matrix_representation


def update_metrics_unsupervised(metrics_n_clust, n_clust, metrics_all_chromosomes):
    '''
    Function updates the dataframe metrics_all_chromosomes with all metrics from one training of a MinCutTAD model for one n_clust.

    :param metrics_n_clust: metrics for unsupervised training for a specific n_clust.
    :param n_clust: number of clusters the TAD regions are supposed to be clustered by the model.
    :param metrics_all_chromosomes: data frame with the metrics for unsupervised training (n_clust, silhouette_score, calinski_harabasz_score, davies_bouldin_score, epocH)
    :return metrics_all_chromosomes:
    '''

    metrics_all_chromosomes = metrics_all_chromosomes.append({"n_clust": n_clust,
                                        "silhouette_score": max(metrics_n_clust["silhouette_score"]),
                                        "calinski_harabasz_score": max(metrics_n_clust["calinski_harabasz_score"]),
                                        "davies_bouldin_score": max(metrics_n_clust["davies_bouldin_score"]),
                                        "epoch": np.where(metrics_n_clust["silhouette_score"] == max(metrics_n_clust["silhouette_score"]))[0][0]
                                        },ignore_index=True)

    return metrics_all_chromosomes

def apply_gnnexplainer(parameters, model, device, dataloader):
    '''
    Function is a wrapper for the functionality of the GNNExplainer (https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.GNNExplainer). It applies to GNNExplainer to all nodes for one specific chromosome, saves the results and creates visualizations for the importance of the node annotations.

    :param parameters: dictionary with parameters set in parameters.json file
    :param model: PyTorch model with trained weights
    :param device: device (cuda/ cpu)
    :param dataloader: dataloader testing
    '''

    explainer = GNNExplainer(model, epochs=parameters["num_epochs_gnn_explanations"], return_type='log_prob')

    node_feat_mask_all = []

    for graph_test_batch in dataloader:
        if graph_test_batch.source_information[0].split("-")[1] != "X":
            X, edge_index, edge_attr, y, source_information = graph_test_batch.x, graph_test_batch.edge_index, graph_test_batch.edge_attr, graph_test_batch.y, graph_test_batch.source_information
            break

    nodes = list(range(X.shape[0]))
    y_nodes = y[nodes].cpu().detach().numpy()

    if parameters["generate_graph_matrix_representation"] == True:
        edge_index = calculation_graph_matrix_representation(parameters, edge_index)
    X, edge_index, edge_attr, y = X.to(device), edge_index.to(device), edge_attr.to(device), y.to(device)

    for node_idx in nodes:
        node_feat_mask, edge_mask = explainer.explain_node(node_idx, x=X, edge_index=edge_index, edge_attr=edge_attr)
        node_feat_mask_all.append(node_feat_mask.cpu().detach().numpy())

        #ax, G = explainer.visualize_subgraph(node_idx, edge_index, edge_mask, y=y)
        #plt.show()

    node_feat_mask_all = np.array(node_feat_mask_all).T

    dict_node_feat_mask_all = {}
    for index, genomic_annotation in enumerate(parameters["genomic_annotations"]):
        dict_node_feat_mask_all[genomic_annotation] = node_feat_mask_all[index]
    dict_node_feat_mask_all['label'] = y_nodes

    df_node_feat_mask_all = pd.DataFrame(data=dict_node_feat_mask_all, index=nodes)

    save_gnn_explanations(parameters, df_node_feat_mask_all)
    visualize_gnnexplanations(parameters, df_node_feat_mask_all)

def visualize_gnnexplanations(parameters, df_node_feat_mask_all):
    '''
    Function generates visualizations of explanations for each node annotation (CTCF, RAD21, SMC3, Housekeeping genes).

    :param parameters: dictionary with parameters set in parameters.json file
    :param df_node_feat_mask_all: dataframe with importance scores for each node annotation (CTCF, RAD21, SMC3, Housekeeping genes) for each node
    '''

    for genomic_annotation in parameters["genomic_annotations"]:
        plt.hist(list(df_node_feat_mask_all[df_node_feat_mask_all['label'] == 0][genomic_annotation]), 50, alpha=0.5, label='No-TAD')
        plt.hist(list(df_node_feat_mask_all[df_node_feat_mask_all['label'] == 1][genomic_annotation]), 50, alpha=0.5, label='TAD')
        min_ylim, max_ylim = plt.ylim()
        plt.axvline(np.mean(list(df_node_feat_mask_all[df_node_feat_mask_all['label'] == 0][genomic_annotation])), color='k', linestyle='dashed', linewidth=1)
        plt.text(np.mean(list(df_node_feat_mask_all[df_node_feat_mask_all['label'] == 0][genomic_annotation])) * 1.01, max_ylim * 0.9, 'Mean: {:.2f}'.format(np.mean(list(df_node_feat_mask_all[df_node_feat_mask_all['label'] == 0][genomic_annotation]))))
        plt.axvline(np.mean(list(df_node_feat_mask_all[df_node_feat_mask_all['label'] == 1][genomic_annotation])), color='k', linestyle='dashed', linewidth=1)
        plt.text(np.mean(list(df_node_feat_mask_all[df_node_feat_mask_all['label'] == 1][genomic_annotation])) * 1.01, max_ylim * 0.8, 'Mean: {:.2f}'.format( np.mean(list(df_node_feat_mask_all[df_node_feat_mask_all['label'] == 1][genomic_annotation]))))
        plt.legend(loc='upper right')
        plt.ylabel("Prevalence")
        plt.xlabel(f"Importance scores for {genomic_annotation}")
        plt.title(f"Explanations for {genomic_annotation}")
        #plt.show()
        plt.savefig(os.path.join(parameters["output_directory"], parameters["dataset_name"], "node_explanations", "node_explanations_distribution_" + genomic_annotation + ".png"))
        plt.close()

    df_node_feat_data = {}

    for genomic_annotation in parameters["genomic_annotations"]:
        df_node_feat_data[genomic_annotation] = [np.mean(df_node_feat_mask_all[df_node_feat_mask_all['label'] == 0][genomic_annotation]), np.mean(df_node_feat_mask_all[df_node_feat_mask_all['label'] == 1][genomic_annotation])]

    df_node_feat = pd.DataFrame(index=["No-TAD", "TAD"], data=df_node_feat_data)
    df_node_feat.plot(kind='bar', title="Mean importance scores of genomic annotations TAD/ No-TAD", ylabel="Mean importance score", rot=0)
    plt.legend(loc='lower right', fontsize="small")
    # plt.show()
    plt.savefig(os.path.join(parameters["output_directory"], parameters["dataset_name"], "node_explanations", "node_explanations_tad_vs_no_tad.png"))
    plt.close()

def save_gnn_explanations(parameters, df_node_feat_mask_all):
    '''
    Function saves explanations by GNNexplainer.

    :param parameters: dictionary with parameters set in parameters.json file
    :param df_node_feat_mask_all: dataframe with importance scores for each node annotation (CTCF, RAD21, SMC3, Housekeeping genes) for each node
    '''

    pd.to_pickle(df_node_feat_mask_all, os.path.join(parameters["output_directory"], parameters["dataset_name"], "node_explanations", "node_explanations.pickle"))

def calculate_classification_metrics(labels_all_chromosomes, scores_all_chromosomes, y_all_chromosomes):
    '''
    Function calculates metrics used for supervised prediction (Accuracy, Precision, AUROC, F1-Score)

    :param y_all_chromosomes: labels for genomic bins generated by model
    :param scores_all_chromosomes: prediction confidence scores for labels for genomic bins generated by model
    :param y_all_chromosomes: true labels for genomic bins
    :return accuracy: accuracy score
    :return precision: precision score
    :return roc_auc: AUROC
    :return f1: F1 score
    '''

    accuracy = accuracy_score(y_all_chromosomes, labels_all_chromosomes)
    precision = precision_score(y_all_chromosomes, labels_all_chromosomes)
    roc_auc = roc_auc_score(y_all_chromosomes, scores_all_chromosomes)
    f1 = f1_score(y_all_chromosomes, labels_all_chromosomes)

    return accuracy, precision, roc_auc, f1

def save_classification_metrics(parameters, scores_all_chromosomes, y_all_chromosomes):
    '''
    Function saves metrics generated by model when using "supervised" as the task type.

    :param parameters: dictionary with parameters set in parameters.json file
    :param scores_all_chromosomes: prediction confidence scores for labels for genomic bins generated by model
    :param y_all_chromosomes: true labels for genomic bins
    '''

    metrics_df = pd.DataFrame(data={'predicted_scores': scores_all_chromosomes, 'true_labels': y_all_chromosomes})
    pd.to_pickle(metrics_df, os.path.join(parameters["output_directory"], parameters["dataset_name"], "metrics", "classification_metrics.pickle"))

def load_classification_metrics(parameters):
    '''
    Function loads metrics generated by model when using "supervised" as the task type.

    :param parameters: dictionary with parameters set in parameters.json file
    :return: metrics generated by model when using "supervised" as the task type (true labels and y label prediction confidence scores)
    '''

    return pd.read_pickle(os.path.join(parameters["output_directory"], parameters["dataset_name"], "metrics", "classification_metrics.pickle"))

def load_classification_metrics_comparison(parameters):
    '''
    Function loads metrics generated by model when using "supervised" as the task type in several experiments and combines the metrics.

    :param parameters: dictionary with parameters set in parameters.json file
    :return: metrics generated by model when using "supervised" as the task type (y label prediction confidence scores)
    '''

    labels_datasets = []
    y_scores_labels_all = []

    for comparison_metric in parameters["comparison_metrics_datasets"]:
        parameters["dataset_name"] = comparison_metric
        comparison_metric_df = load_classification_metrics(parameters)

        labels_datasets.append(list(comparison_metric_df['true_labels']))
        y_scores_labels_all.append(list(comparison_metric_df['predicted_scores']))

    return labels_datasets, y_scores_labels_all

def save_metrics_all_n_clust(parameters, metrics_df, type_metrics):
    '''
    Function saves metrics generated by model when using "unsupervised" as the task type for all n_clust.

    :param parameters: dictionary with parameters set in parameters.json file
    :param type_metrics: indicator given, when cross-validation is taking place
    :param metrics generated by model when using "unsupervised" as the task type for all n_clust:
    '''

    pd.to_pickle(metrics_df, os.path.join(parameters["output_directory"], parameters["dataset_name"], "metrics",
                                          "metrics_" + type_metrics + ".pickle"))

def load_metrics_all_n_clust(parameters, type_metrics):
    '''
    Function loads metrics generated by model when using "unsupervised" as the task type for all n_clust.

    :param parameters: dictionary with parameters set in parameters.json file
    :param type_metrics: indicator given, when cross-validation is taking place
    :return metrics_df: metrics generated by model when using "unsupervised" as the task type for all n_clust
    '''

    metrics_df = pd.read_pickle(os.path.join(parameters["output_directory"], parameters["dataset_name"], "metrics",
                                          "metrics_" + type_metrics + ".pickle"))

    return metrics_df


def load_unsupervised_metrics_comparison(parameters):
    '''
    Function loads metrics generated by model when using "unsupervised" as the task type in several experiments and combines the metrics.

    :param parameters: dictionary with parameters set in parameters.json file
    :return comparison_metric_df_all: dataframe with combined metric from different experiments
    '''

    comparison_metric_df_all = pd.DataFrame()

    for comparison_metric, comparison_metric_label in zip(parameters["comparison_metrics_datasets"], parameters["comparison_metrics_datasets_labels"]):
        parameters["dataset_name"] = comparison_metric
        comparison_metric_df = load_metrics_all_n_clust(parameters, "train_all_n_clust")
        #This assumes that all models are iterating over the same n_clust.
        if "n_clust" not in comparison_metric_df_all.columns:
            comparison_metric_df_all[f'n_clust'] = comparison_metric_df['n_clust']
        comparison_metric_df_all[f'Silhouette score: {comparison_metric_label}'] = comparison_metric_df['silhouette_score']

    return comparison_metric_df_all

def save_metrics(parameters, metrics, type_metrics, n_clust=None):
    '''
    Function saves metrics generated by model.

    :param parameters: dictionary with parameters set in parameters.json file
    :param metrics: dict with metrics generated by model
    :param type_metrics: indicator given, when cross-validation is taking place
    :param n_clust: number of clusters
    '''

    metrics_df = pd.DataFrame(data=metrics)
    if parameters["task_type"] == "supervised":
        pd.to_pickle(metrics_df, os.path.join(parameters["output_directory"], parameters["dataset_name"], "metrics",
                                          "metrics_" + type_metrics + ".pickle"))
    elif parameters["task_type"] == "unsupervised":
        pd.to_pickle(metrics_df, os.path.join(parameters["output_directory"], parameters["dataset_name"], "metrics",
                                          "metrics_" + str(n_clust) + "_" + type_metrics + ".pickle"))

def load_metrics(parameters, type_metrics):
    '''
    Function loads metrics generated by model.

    :param parameters: dictionary with parameters set in parameters.json file
    :param type_metrics: indicator given, when cross-validation is taking place
    :return metrics: dataframe with metrics generated by model
    '''

    return pd.read_pickle(os.path.join(parameters["output_directory"], parameters["dataset_name"], "metrics",
                                          "metrics_" + type_metrics + ".pickle"))

def setup_metrics(parameters):
    '''
    Function sets up metrics calculated during training for supervised and unsupervised task type.

    :param parameters: dictionary with parameters set in parameters.json file
    :return metrics: dictionary with metrics as keys, gets updated during training
    '''

    if parameters["task_type"] == "supervised":
        metrics = {
            "silhouette_score": [],
            "calinski_harabasz_score": [],
            "davies_bouldin_score": [],
            "homogeneity_score": [],
            "completeness_score": [],
            "v_measure_score": [],
            "computation_time": [],
            "accuracy": [],
            "precision": [],
            "roc_auc": [],
            "f1": []
        }
    elif parameters["task_type"] == "unsupervised":
        metrics = {
            "n_clust": [],
            "silhouette_score": [],
            "calinski_harabasz_score": [],
            "davies_bouldin_score": [],
            "computation_time": []
        }

    return metrics

def generate_metrics_plots(parameters, metrics_all_chr, type):
    '''
    Function generates plots for each score metric in score_metrics_clustering for deviating number of clusters in MinCutTAD.

    :param parameters: dictionary with parameters set in parameters.json file
    :param metrics_all_chr: dataframe with score metrics determined for different number of clusters
    :param type: indicator whether comparison of two different experiments
    '''

    if "vs" not in type:
        for metric in list(metrics_all_chr.columns)[1:]:
            plt.plot("n_clust", metric, data=metrics_all_chr)
            plt.xlabel("Number clusters")
            plt.ylabel(metric)
            #plt.show()
            plt.savefig(os.path.join(parameters["output_directory"], parameters["dataset_name"], "plots", type + "_" + metric + "_vs_n_clust.png"))
            plt.close()

    labels_metrics = []
    for metric in list(metrics_all_chr.columns)[1:]:
        plt.plot("n_clust", metric, data=metrics_all_chr)
        labels_metrics += [metric]
    plt.xlabel("Number clusters")
    if "vs" not in type:
        plt.ylabel("Values of metrics")
    else:
        plt.ylabel("Silhouette score")
    plt.legend(labels=labels_metrics)
    #plt.show()
    plt.savefig(os.path.join(parameters["output_directory"], parameters["dataset_name"], "plots", type + "_" + "all_metrics_vs_n_clust.png"))
    plt.close()

def choose_optimal_n_clust(metrics_train_all_chromosomes):
    '''
    Function determines optimal number of clusters for MinCutTAD prediction based on max silhouette score.

    :param metrics_train_all_chromosomes: dataframe with maximal silhouette scores for deviating n_clust.
    :return optimal_n_clust: optimal number of clusters for MinCutTAD prediction
    '''

    silhouette_scores_list = []
    n_clust_list = []

    for n_clust_metrics_train_all_chromosomes in np.unique(metrics_train_all_chromosomes['n_clust']):
        n_clust_list.append(n_clust_metrics_train_all_chromosomes)
        silhouette_scores_list.append(float(max(metrics_train_all_chromosomes[metrics_train_all_chromosomes['n_clust'] == n_clust_metrics_train_all_chromosomes]["silhouette_score"])))

    index_n_clust_epoch = np.where(np.array(silhouette_scores_list) == max(silhouette_scores_list))[0][0]
    optimal_n_clust = n_clust_list[index_n_clust_epoch]
    optimal_epoch = np.where(np.array(metrics_train_all_chromosomes[metrics_train_all_chromosomes["n_clust"] == optimal_n_clust]['silhouette_score']) == max(silhouette_scores_list))[0][0]

    return optimal_n_clust, optimal_epoch

def metrics_calculation_unsupervised(X, edge_index, edge_attr, labels):
    '''
    Function calculates a range of metrics based on clustering quality (Methods: Silhouette score,  Silhouette samples score, Calinski Harabasz score, Davies bouldin score) and comparison of labels called by MinCutTAD (labels) and labels called by Arrowhead (labels_true) (Methods: Homogeneity score, Completeness score, V-measure score).

    :param edge_index: edge index representation of adjacency_matrix, reports the nodes and the edges between the nodes present in a graph
    :param edge_attr: edge attributes representation of adjacency_matrix, reports the attributes, e.g. the weight, of each edge in edge_index
    :param labels: labels of genomic bins in adjacency matrix called by MinCutTAD model
    :return silhouette_score_calc: Silhouette score based on clustering quality of adjacency matrix.
    :return calinski_harabasz_score_calc: Calinski Harabasz score based on clustering quality of adjacency matrix.
    :return davies_bouldin_score_calc: Davies bouldin score based on clustering quality of adjacency matrix.
    '''

    adj = to_dense_adj(edge_index=edge_index, edge_attr=edge_attr)[0]

    if len(np.unique(labels)) > 1:
        silhouette_score_calc = silhouette_score(adj.cpu().numpy(), labels)  # The Silhouette Coefficient is calculated using the mean intra-cluster distance (a) and the mean nearest-cluster distance (b) for each sample. The Silhouette Coefficient for a sample is (b - a) / max(a, b). To clarify, b is the distance between a sample and the nearest cluster that the sample is not a part of. Note that Silhouette Coefficient is only defined if number of labels is 2 <= n_labels <= n_samples - 1. The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters. Negative values generally indicate that a sample has been assigned to the wrong cluster, as a different cluster is more similar.
        # Xarray-like of shape (n_samples_a, n_samples_a) if metric == “precomputed” or (n_samples_a, n_features) otherwise - An array of pairwise distances between samples, or a feature array.
        calinski_harabasz_score_calc = calinski_harabasz_score(X.cpu(), labels)  # The score is defined as ratio between the within-cluster dispersion and the between-cluster dispersion.
        # X-array-like of shape (n_samples, n_features) - A list of n_features-dimensional data points. Each row corresponds to a single data point.
        davies_bouldin_score_calc = davies_bouldin_score(X.cpu(), labels)  # The score is defined as the average similarity measure of each cluster with its most similar cluster, where similarity is the ratio of within-cluster distances to between-cluster distances. Thus, clusters which are farther apart and less dispersed will result in a better score. The minimum score is zero, with lower values indicating better clustering.
        # X-array-like of shape (n_samples, n_features) - A list of n_features-dimensional data points. Each row corresponds to a single data point.

        return silhouette_score_calc, calinski_harabasz_score_calc, davies_bouldin_score_calc
    else:
        print("No positive labels have been predicted.")

        return 0, 0, 0

def metrics_calculation_supervised(labels, labels_true):
    '''
    Function calculates a range of metrics based on clustering quality (Methods: Silhouette score,  Silhouette samples score, Calinski Harabasz score, Davies bouldin score) and comparison of labels called by MinCutTAD (labels) and labels called by Arrowhead (labels_true) (Methods: Homogeneity score, Completeness score, V-measure score).

    :param labels: labels of genomic bins in all adjacency matrices called by MinCutTAD model
    :param labels_true: labels of genomic bins in all adjacency matrices called by Arrowhead (Ground Truth)
    :return homogeneity_score_calc: Homogeneity score based on comparison labels and labels_true
    :return completeness_score_calc: Completeness score based on comparison labels and labels_true
    :return v_measure_score_calc: V-measure score based on comparison labels and labels_true
    '''

    homogeneity_score_calc = homogeneity_score(labels, labels_true)  # A clustering result satisfies homogeneity if all of its clusters contain only data points which are members of a single class. This metric is independent of the absolute values of the labels: a permutation of the class or cluster label values won’t change the score value in any way.
    completeness_score_calc = completeness_score(labels, labels_true)  # A clustering result satisfies completeness if all the data points that are members of a given class are elements of the same cluster. - This metric is independent of the absolute values of the labels: a permutation of the class or cluster label values won’t change the score value in any way.
    v_measure_score_calc = v_measure_score(labels, labels_true)  # The V-measure is the harmonic mean between homogeneity and completeness: v = (1 + beta) * homogeneity * completeness / (beta * homogeneity + completeness)

    return homogeneity_score_calc, completeness_score_calc, v_measure_score_calc


def plot_roc_curve(parameters, labels_datasets, y_scores_labels_all, labels_comparison_datasets=None):
    '''
    Function generates ROC curves based on labels and scores in labels_datasets and y_scores_labels_all.

    :param parameters: dictionary with parameters set in parameters.json file
    :param labels_dataset: true labels for one or more datasets
    :param y_scores_labels_all: prediction confidence scores by dl or ml model for one or more datasets
    :param labels_comparison_datasets: names of datasets for comparison
    '''

    plt.rcParams["figure.figsize"] = (5, 5)
    xlabel_text = ""
    for index_scores_labels, (y_scores_labels, label_dataset) in enumerate(zip(y_scores_labels_all, labels_datasets)):
        fpr, tpr, _ = metrics.roc_curve(np.array(label_dataset).astype("int"), np.array(y_scores_labels))
        roc_auc = metrics.auc(fpr, tpr)
        ax = sns.lineplot(x=fpr, y=tpr)
        if labels_comparison_datasets:
            xlabel_text += f'AUROC {labels_comparison_datasets[index_scores_labels]} {roc_auc:.2f}     '
        else:
            xlabel_text += f'AUROC {roc_auc:.2f}'

    ax.set_xlabel(xlabel_text)
    ax.plot([0, 1], [0, 1], c='grey')
    if labels_comparison_datasets:
        ax.legend(loc='lower right', labels=labels_comparison_datasets)

    #plt.title(parameters["datasets"]["phenotypes_plot"])
    plt.tight_layout()

    if labels_comparison_datasets:
        labels_datasets_name = "_".join(labels_comparison_datasets)
        plt.savefig(os.path.join(parameters["output_directory"], parameters["dataset_name"], "plots", "roc_" + labels_datasets_name + ".png"))
    else:
        plt.savefig(os.path.join(parameters["output_directory"], parameters["dataset_name"], "plots", "roc.png"))

    plt.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_parameters_json", help=" to JSON with parameters.", type=str, default='../tad_detection/model/parameters.json')
    args = parser.parse_args()
    path_parameters_json = args.path_parameters_json
    # path_parameters_json = "./tad_detection/model/parameters.json"

    parameters = load_parameters(path_parameters_json)
    os.makedirs(parameters["output_directory"], exist_ok=True)
    os.makedirs(os.path.join(parameters["output_directory"], parameters["comparison_metrics_name"]), exist_ok=True)
    os.makedirs(os.path.join(parameters["output_directory"], parameters["comparison_metrics_name"], "plots"), exist_ok=True)
    dump_parameters(parameters)

    logger = set_up_logger('metrics_comparison', parameters)
    logger.debug('Start metrics_comparison logger.')

    if parameters["task_type"] == "supervised":
        labels_datasets, y_scores_labels_all = load_classification_metrics_comparison(parameters)
        plot_roc_curve(parameters, labels_datasets, y_scores_labels_all, labels_comparison_datasets=parameters["comparison_metrics_datasets_labels"])
    elif parameters["task_type"] == "unsupervised":
        metrics_comparison = load_unsupervised_metrics_comparison(parameters)
        generate_metrics_plots(parameters, metrics_comparison, parameters["comparison_metrics_name"])

    else:
        raise NotImplementedError("Please choose 'supervised' or 'unsupervised' as your 'task_type'.")

