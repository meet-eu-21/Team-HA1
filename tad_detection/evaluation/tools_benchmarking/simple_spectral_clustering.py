import sys
sys.path.insert(1, './tad_detection/')
sys.path.insert(1, './tad_detection/preprocessing/')
sys.path.insert(1, './tad_detection/model/')
sys.path.insert(1, './tad_detection/evaluation/')

from sklearn.cluster import spectral_clustering
from utils_general import load_parameters, set_up_logger
from utils_model import load_data, generate_metrics_plots, choose_optimal_n_clust, metrics_calculation, save_tad_list
import pandas as pd
import os
import time
import argparse

def train(data, parameters):

    n_clust = 0

    n_clust_list = []

    silhouette_score_list_baseline = []
    silhouette_samples_list_baseline = []
    homogeneity_score_list_baseline = []
    completeness_score_list_baseline = []
    v_measure_score_list_baseline = []
    calinski_harabasz_score_list_baseline = []
    davies_bouldin_score_list_baseline = []
    time_list_baseline = []

    # Data
    X = data.X
    edge_index = data.edge_index
    labels_true = data.y

    while n_clust < 50:
        n_clust += 1
        n_clust_list.append(n_clust)

        if len(silhouette_score_list_baseline) > 3 and np.all(silhouette_score_list_baseline[-1:-3] < max(silhouette_score_list_baseline)):
            print("Clustering with " + str(n_clust) + " clusters.")

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
            print("A maximum silhouette score has been detected. Further clustering has been stopped. Evaluation scripts are now generated.")
            break

    score_metrics_clustering_baseline = pd.DataFrame(list(zip(n_clust_list, silhouette_score_list_baseline, silhouette_samples_list_baseline, homogeneity_score_list_baseline, completeness_score_list_baseline, davies_bouldin_score_list_baseline, calinski_harabasz_score_list_baseline, davies_bouldin_score_list_baseline)),
                   columns =["Number clusters", "Calculation time algorithm", "Silhouette score", "Silhouette scores of genomic locations (bins)", "Homogeneity score", "Completeness score", "V measure score", "Calinski Harabasz score", "Davies Bouldin score"])
    score_metrics_clustering_baseline.to_pickle(os.path.join(parameters["output_directory"], "score_metrics_clustering_baseline.pickle"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_parameters_json", help=" to JSON with parameters.", type=str, required=True)
    args = parser.parse_args()
    path_parameters_json = args.path_parameters_json

    parameters = load_parameters(path_parameters_json)
    os.makedirs(parameters["output_directory"], exist_ok=True)
    os.makedirs(os.path.join(parameters["output_directory"], "training"), exist_ok=True)
    logger = set_up_logger('simple_spectral_clustering', parameters)
    logger.debug('Start simple_spectral_clustering logger.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = load_data(parameters, device)
    score_metrics_clustering_baseline, predicted_tad = train(data, parameters)

    generate_metrics_plots(score_metrics_clustering_baseline)

    optimal_n_clust = choose_optimal_n_clust()

    save_tad_list(parameters, predicted_tad[optimal_n_clust], "Spectral")
