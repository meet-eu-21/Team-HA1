import sys
sys.path.insert(1, './tad_detection/')
sys.path.insert(1, './tad_detection/preprocessing/')
sys.path.insert(1, './tad_detection/model/')
sys.path.insert(1, './tad_detection/evaluation/')

import numpy as np
from utils_general import load_parameters, dump_parameters, set_up_logger
from model.utils_model import set_up_neptune, load_data, split_data, torch_geometric_data_generation_dataloader, load_optimizer, save_model, load_model, calculation_graph_matrix_representation, save_tad_list, determine_tad_regions
from model.metrics import setup_metrics, generate_metrics_plots, choose_optimal_n_clust, metrics_calculation_supervised, metrics_calculation_unsupervised, calculate_classification_metrics, save_metrics, save_metrics_all_n_clust, apply_gnnexplainer, save_classification_metrics, plot_roc_curve, update_metrics_unsupervised
from model.mincuttad import MinCutTAD
import pandas as pd
import os
import argparse
import torch
import torch.nn.functional as F
import time
from collections import Counter
from torch_geometric.loader import DataLoader
from torch.autograd import Variable

def train_wrapper(parameters, dataloader_train_cross_1, dataloader_val_cross_1, dataloader_train_cross_2, dataloader_val_cross_2):

    if parameters["task_type"] == "unsupervised":

        metrics_train_all_chromosomes = pd.DataFrame(columns=["n_clust", "silhouette_score", "calinski_harabasz_score", "davies_bouldin_score", "epoch"])
        metrics_valid_all_chromosomes = pd.DataFrame(columns=["n_clust", "silhouette_score", "calinski_harabasz_score", "davies_bouldin_score", "epoch"])
        n_clust = 5

        while n_clust < 1000:

            metrics_train = setup_metrics(parameters)
            metrics_valid = setup_metrics(parameters)

            n_clust += 1

            model = MinCutTAD(parameters, n_clust).to(device)
            optimizer, scheduler = load_optimizer(model, parameters)


            if not isinstance(dataloader_train_cross_2, DataLoader):
                logger.info("Training with one cell line, so no cross validation is performed.")
                model, metrics_train, metrics_valid = train(parameters, model, dataloader_train_cross_1, dataloader_val_cross_1,
                                                                optimizer, scheduler, device, logger, metrics_train,
                                                                metrics_valid, run, n_clust)

                save_metrics(parameters, metrics_train, "train", n_clust=n_clust)
                save_metrics(parameters, metrics_valid, "valid", n_clust=n_clust)

                metrics_train_all_chromosomes = update_metrics_unsupervised(metrics_train, n_clust, metrics_train_all_chromosomes)
                metrics_valid_all_chromosomes = update_metrics_unsupervised(metrics_valid, n_clust, metrics_valid_all_chromosomes)

            else:
                logger.info("Cross validation between cell lines GM12878 and IMR-90.")
                model_cross_1, metrics_train_cross_1, metrics_valid_cross_1, model_cross_2, metrics_train_cross_2, metrics_valid_cross_2 = train_cross(
                    parameters, model, dataloader_train_cross_1, dataloader_val_cross_1, dataloader_train_cross_2,
                    dataloader_val_cross_2, optimizer, scheduler, device, logger, metrics_train, metrics_valid, run, n_clust)

                save_metrics(parameters, metrics_train_cross_1, "train_cross_1", n_clust=n_clust)
                save_metrics(parameters, metrics_train_cross_2, "train_cross_2", n_clust=n_clust)
                save_metrics(parameters, metrics_valid_cross_1, "valid_cross_1", n_clust=n_clust)
                save_metrics(parameters, metrics_valid_cross_2, "valid_cross_2", n_clust=n_clust)

                metrics_train_all_chromosomes = update_metrics_unsupervised(metrics_train_cross_1, n_clust, metrics_train_all_chromosomes)
                metrics_valid_all_chromosomes = update_metrics_unsupervised(metrics_valid_cross_1, n_clust, metrics_valid_all_chromosomes)

            silhouette_scores_metrics_train_all_chromosomes_list = []
            for n_clust_metrics_train_all_chromosomes in np.unique(metrics_train_all_chromosomes['n_clust']):
                silhouette_scores_metrics_train_all_chromosomes_list.append(float(max(metrics_train_all_chromosomes[metrics_train_all_chromosomes['n_clust'] == n_clust_metrics_train_all_chromosomes]["silhouette_score"])))

            if (len(metrics_train_all_chromosomes) < 5) or (len(metrics_train_all_chromosomes) > 3 and all(silhouette_score_eval < max(silhouette_scores_metrics_train_all_chromosomes_list) for silhouette_score_eval in silhouette_scores_metrics_train_all_chromosomes_list[-5:-1])):
                logger.info("Clustering with " + str(n_clust) + " clusters.")

            else:
                logger.info(
                    "A maximum silhouette score has been detected. Further clustering has been stopped. Evaluation scripts are now generated.")
                break

        save_metrics_all_n_clust(parameters, metrics_train_all_chromosomes, "train_all_n_clust")
        save_metrics_all_n_clust(parameters, metrics_train_all_chromosomes, "valid_all_n_clust")

        generate_metrics_plots(parameters, metrics_train_all_chromosomes, "train_all_n_clust")
        generate_metrics_plots(parameters, metrics_valid_all_chromosomes, "valid_all_n_clust")

        optimal_n_clust, optimal_epoch = choose_optimal_n_clust(metrics_train_all_chromosomes)
        model_for_test = load_model(parameters, optimal_epoch, n_clust=optimal_n_clust)

    elif parameters["task_type"] == "supervised":

        metrics_train = setup_metrics(parameters)
        metrics_valid = setup_metrics(parameters)

        model = MinCutTAD(parameters).to(device)
        optimizer, scheduler = load_optimizer(model, parameters)

        if not isinstance(dataloader_train_cross_2, DataLoader):
            logger.info("Training with one cell line, so no cross validation is performed.")
            model, metrics_train, metrics_valid = train(parameters, model, dataloader_train_cross_1,
                                                            dataloader_val_cross_1,
                                                            optimizer, scheduler, device, logger, metrics_train,
                                                            metrics_valid, run)

            save_metrics(parameters, metrics_train, "train")
            save_metrics(parameters, metrics_valid, "valid")

        else:
            logger.info("Cross validation between cell lines GM12878 and IMR-90.")
            model_cross_1, metrics_train_cross_1, metrics_valid_cross_1, model_cross_2, metrics_train_cross_2, metrics_valid_cross_2 = train_cross(
                parameters, model, dataloader_train_cross_1, dataloader_val_cross_1, dataloader_train_cross_2,
                dataloader_val_cross_2, optimizer, scheduler, device, logger, metrics_train, metrics_valid, run)

            save_metrics(parameters, metrics_train_cross_1, "train_cross_1")
            save_metrics(parameters, metrics_train_cross_2, "train_cross_2")
            save_metrics(parameters, metrics_valid_cross_1, "valid_cross_1")
            save_metrics(parameters, metrics_valid_cross_2, "valid_cross_2")

        try:
            optimal_epoch = np.where(metrics_train_cross_1["roc_auc"] == max(metrics_train_cross_1["roc_auc"]))[0]
        except:
            optimal_epoch = np.where(metrics_train["roc_auc"] == max(metrics_train["roc_auc"]))[0]
        model_for_test = load_model(parameters, optimal_epoch)

    else:
        NotImplementedError("Please choose 'unsupervised' or 'supervised' as the task type.")

    return model_for_test

def train_cross(parameters, model, dataloader_train_cross_1, dataloader_val_cross_1, dataloader_train_cross_2, dataloader_val_cross_2, optimizer, scheduler, device, logger, metrics_train, metrics_valid, run, n_clust=None):

    logger.info("Training of cross-validation 1")
    model_cross_1, metrics_train_cross_1, metrics_valid_cross_1 = train(parameters, model, dataloader_train_cross_1, dataloader_val_cross_1,
                                                    optimizer, scheduler, device, logger, metrics_train, metrics_valid, run, n_clust=n_clust)

    logger.info("Training of cross-validation 2")
    model_cross_2, metrics_train_cross_2, metrics_valid_cross_2 = train(parameters, model, dataloader_train_cross_2, dataloader_val_cross_2,
                                                    optimizer, scheduler, device, logger, metrics_train, metrics_valid, run, n_clust=n_clust)

    return model_cross_1, metrics_train_cross_1, metrics_valid_cross_1, model_cross_2, metrics_train_cross_2, metrics_valid_cross_2

def train(parameters, model, dataloader_train, dataloader_val, optimizer, scheduler, device, logger, metrics_train, metrics_valid, run, n_clust=None):

    for epoch in range(0, parameters["epoch_num"]):

        start_time_mincuttad = time.time()
        labels_all_chromosomes = np.array([])
        y_all_chromosomes = np.array([])
        scores_all_chromosomes = np.array([])

        silhouette_scores_chromosomes = []
        calinski_harabasz_scores_chromosomes = []
        davies_bouldin_scores_chromosomes = []

        for graph_train_batch in dataloader_train:

            model.train()

            X, edge_index, edge_attr, y = graph_train_batch.x, graph_train_batch.edge_index, graph_train_batch.edge_attr, graph_train_batch.y
            if parameters["generate_graph_matrix_representation"] == True:
                edge_index = calculation_graph_matrix_representation(parameters, edge_index)
            X, edge_index, edge_attr, y = X.to(device), edge_index.to(device), edge_attr.to(device), y.to(device)
            #x - torch.Size([2493, 4]), edge_index -torch.Size([2, 4666109]), edge_attr - torch.Size([4666109]), y - torch.Size([2493])

            optimizer.zero_grad()

            out, lp_loss, entropy_loss = model(X, edge_index, edge_attr)

            labels = np.argmax(out.cpu().detach().numpy(), axis=-1)
            print(Counter(labels))

            if parameters["task_type"] == "supervised":
                loss = F.nll_loss(out, y.view(-1).long(), reduction='mean')
            elif parameters["task_type"] == "unsupervised":
                loss = lp_loss + entropy_loss
                loss = Variable(torch.Tensor([loss]), requires_grad=True)

            # run["logs/training/batch/loss"].log(loss)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            run["loss_train"].log(loss)

            save_model(model, epoch, n_clust, parameters)

            silhouette_score, calinski_harabasz_score, davies_bouldin_score = metrics_calculation_unsupervised(X, edge_index, edge_attr, labels)
            silhouette_scores_chromosomes.append(silhouette_score)
            calinski_harabasz_scores_chromosomes.append(calinski_harabasz_score)
            davies_bouldin_scores_chromosomes.append(davies_bouldin_score)

            labels_all_chromosomes = np.concatenate((labels_all_chromosomes, labels))
            y_all_chromosomes = np.concatenate((y_all_chromosomes, y.cpu().numpy()))
            scores_all_chromosomes = np.concatenate((scores_all_chromosomes, out.cpu().detach().numpy().T[0]))

        end_time_mincuttad = time.time()

        run["computation_time_train"].log(end_time_mincuttad - start_time_mincuttad)
        metrics_train["computation_time"].append(end_time_mincuttad - start_time_mincuttad)

        if parameters["task_type"] == "supervised":
            homogeneity_score, completeness_score, v_measure_score = metrics_calculation_supervised(labels_all_chromosomes, y_all_chromosomes)
            run["homogeneity_score_train"].log(homogeneity_score)
            run["completeness_score_train"].log(completeness_score)
            run["v_measure_score_train"].log(v_measure_score)
            metrics_train["homogeneity_score"].append(homogeneity_score)
            metrics_train["completeness_score"].append(completeness_score)
            metrics_train["v_measure_score"].append(v_measure_score)
            accuracy_score, precision_score, roc_auc_score, f1_score = calculate_classification_metrics(labels_all_chromosomes, scores_all_chromosomes, y_all_chromosomes)
            run["accuracy_train"].log(accuracy_score)
            run["precision_train"].log(precision_score)
            run["roc_auc_train"].log(roc_auc_score)
            run["f1_train"].log(f1_score)
            metrics_train["accuracy"].append(accuracy_score)
            metrics_train["precision"].append(precision_score)
            metrics_train["roc_auc"].append(roc_auc_score)
            metrics_train["f1"].append(f1_score)
        elif parameters["task_type"] == "unsupervised":
            run["n_clust_train"].log(n_clust)
            metrics_train["n_clust"].append(n_clust)
        run["silhouette_score_train"].log(np.mean(silhouette_scores_chromosomes))
        run["calinski_harabasz_score_train"].log(np.mean(calinski_harabasz_scores_chromosomes))
        run["davies_bouldin_score_train"].log(np.mean(davies_bouldin_scores_chromosomes))
        metrics_train["silhouette_score"].append(np.mean(silhouette_scores_chromosomes))
        metrics_train["calinski_harabasz_score"].append(np.mean(calinski_harabasz_scores_chromosomes))
        metrics_train["davies_bouldin_score"].append(np.mean(davies_bouldin_scores_chromosomes))

        #print("Evaluation")
        with torch.no_grad():

            start_time_mincuttad = time.time()
            labels_all_chromosomes = np.array([])
            y_all_chromosomes = np.array([])
            scores_all_chromosomes = np.array([])

            silhouette_scores_chromosomes = []
            calinski_harabasz_scores_chromosomes = []
            davies_bouldin_scores_chromosomes = []

            for graph_val_batch in dataloader_val:

                model.eval()

                X, edge_index, edge_attr, y = graph_val_batch.x, graph_val_batch.edge_index, graph_val_batch.edge_attr, graph_val_batch.y
                if parameters["generate_graph_matrix_representation"] == True:
                    edge_index = calculation_graph_matrix_representation(parameters, edge_index)
                X, edge_index, edge_attr, y = X.to(device), edge_index.to(device), edge_attr.to(device), y.to(device)

                out, lp_loss, entropy_loss = model(X, edge_index, edge_attr)

                labels = np.argmax(out.cpu().detach().numpy(), axis=-1)
                print(Counter(labels))

                if parameters["task_type"] == "supervised":
                    loss = F.nll_loss(out, y.view(-1).long(), reduction='mean')
                elif parameters["task_type"] == "unsupervised":
                    loss = lp_loss + entropy_loss
                    loss = Variable(torch.Tensor([loss]), requires_grad=True)

                run["loss_valid"].log(loss)

                silhouette_score, calinski_harabasz_score, davies_bouldin_score = metrics_calculation_unsupervised(X, edge_index, edge_attr, labels)
                silhouette_scores_chromosomes.append(silhouette_score)
                calinski_harabasz_scores_chromosomes.append(calinski_harabasz_score)
                davies_bouldin_scores_chromosomes.append(davies_bouldin_score)

                labels_all_chromosomes = np.concatenate((labels_all_chromosomes, labels))
                y_all_chromosomes = np.concatenate((y_all_chromosomes, y.cpu().numpy()))
                scores_all_chromosomes = np.concatenate((scores_all_chromosomes, out.cpu().detach().numpy().T[0]))

            end_time_mincuttad = time.time()

            run["computation_time_valid"].log(end_time_mincuttad - start_time_mincuttad)
            metrics_valid["computation_time"].append(end_time_mincuttad - start_time_mincuttad)

            if parameters["task_type"] == "supervised":
                homogeneity_score, completeness_score, v_measure_score = metrics_calculation_supervised(labels_all_chromosomes, y_all_chromosomes)
                run["homogeneity_score_valid"].log(homogeneity_score)
                run["completeness_score_valid"].log(completeness_score)
                run["v_measure_score_valid"].log(v_measure_score)
                metrics_valid["homogeneity_score"].append(homogeneity_score)
                metrics_valid["completeness_score"].append(completeness_score)
                metrics_valid["v_measure_score"].append(v_measure_score)
                accuracy_score, precision_score, roc_auc_score, f1_score = calculate_classification_metrics(labels_all_chromosomes, scores_all_chromosomes, y_all_chromosomes)
                run["accuracy_valid"].log(accuracy_score)
                run["precision_valid"].log(precision_score)
                run["roc_auc_valid"].log(roc_auc_score)
                run["f1_valid"].log(f1_score)
                metrics_valid["accuracy"].append(accuracy_score)
                metrics_valid["precision"].append(precision_score)
                metrics_valid["roc_auc"].append(roc_auc_score)
                metrics_valid["f1"].append(f1_score)
            elif parameters["task_type"] == "unsupervised":
                run["n_clust_valid"].log(n_clust)
                metrics_valid["n_clust"].append(n_clust)
            run["silhouette_score_valid"].log(np.mean(silhouette_scores_chromosomes))
            run["calinski_harabasz_score_valid"].log(np.mean(calinski_harabasz_scores_chromosomes))
            run["davies_bouldin_score_valid"].log(np.mean(davies_bouldin_scores_chromosomes))
            metrics_valid["silhouette_score"].append(np.mean(silhouette_scores_chromosomes))
            metrics_valid["calinski_harabasz_score"].append(np.mean(calinski_harabasz_scores_chromosomes))
            metrics_valid["davies_bouldin_score"].append(np.mean(davies_bouldin_scores_chromosomes))

            scheduler.step(loss)

    return model, metrics_train, metrics_valid


def final_validation_cross(parameters, model_cross_1, model_cross_2, dataloader_test_cross_1, dataloader_test_cross_2, device, logger, run):

    logger.info("Final validation of cross-validation 1")
    parameters["cross_run"] = "cross_run_1"
    predicted_tads_cross_1 = final_validation(parameters, model_cross_1, dataloader_test_cross_1, device, logger, run)

    logger.info("Final validation of cross-validation 2")
    parameters["cross_run"] = "cross_run_2"
    predicted_tads_cross_2 = final_validation(parameters, model_cross_2, dataloader_test_cross_2, device, logger, run)

    return predicted_tads_cross_1, predicted_tads_cross_2

def final_validation(parameters, model, dataloader_test, device, logger, run):

    metrics_test = setup_metrics(parameters)

    with torch.no_grad():

        start_time_mincuttad = time.time()
        labels_all_chromosomes = np.array([])
        y_all_chromosomes = np.array([])
        scores_all_chromosomes = np.array([])

        silhouette_scores_chromosomes = []
        calinski_harabasz_scores_chromosomes = []
        davies_bouldin_scores_chromosomes = []
        predicted_tads = {}
        cell_line = list(dataloader_test)[0].source_information[0].split("-")[0]
        predicted_tads[cell_line] = {}

        for graph_test_batch in dataloader_test:

            model.eval()

            X, edge_index, edge_attr, y, source_information = graph_test_batch.x, graph_test_batch.edge_index, graph_test_batch.edge_attr, graph_test_batch.y, graph_test_batch.source_information
            if parameters["generate_graph_matrix_representation"] == True:
                edge_index = calculation_graph_matrix_representation(parameters, edge_index)
            X, edge_index, edge_attr, y = X.to(device), edge_index.to(device), edge_attr.to(device), y.to(device)

            out, lp_loss, entropy_loss = model(X, edge_index, edge_attr)

            labels = np.argmax(out.cpu().detach().numpy(), axis=-1)
            print(Counter(labels))
            if parameters["task_type"] == "supervised":
                predicted_tads[cell_line][source_information[0].split("-")[1]] = np.where(labels == 0)[0]
            elif parameters["task_type"] == "unsupervised":
                tad_group_chromosome = []
                for tad_group in [x for _, x in sorted(zip(list(Counter(labels).values()), list(Counter(labels).keys())))][::-1][1:]:
                    tad_group_chromosome.append(np.where(labels == tad_group)[0])
                    predicted_tads[cell_line][source_information[0].split("-")[1]] = np.array(tad_group_chromosome)

            if parameters["task_type"] == "supervised":
                loss = F.nll_loss(out, y.view(-1).long(), reduction='mean')
            elif parameters["task_type"] == "unsupervised":
                loss = lp_loss + entropy_loss
                loss = Variable(torch.Tensor([loss]), requires_grad=True)

            run["loss_test"].log(loss)

            silhouette_score, calinski_harabasz_score, davies_bouldin_score = metrics_calculation_unsupervised(X, edge_index, edge_attr, labels)
            silhouette_scores_chromosomes.append(silhouette_score)
            calinski_harabasz_scores_chromosomes.append(calinski_harabasz_score)
            davies_bouldin_scores_chromosomes.append(davies_bouldin_score)

            labels_all_chromosomes = np.concatenate((labels_all_chromosomes, labels))
            y_all_chromosomes = np.concatenate((y_all_chromosomes, y.cpu().numpy()))
            scores_all_chromosomes = np.concatenate((scores_all_chromosomes, out.cpu().detach().numpy().T[0]))

        end_time_mincuttad = time.time()

        run["computation_time_valid"].log(end_time_mincuttad - start_time_mincuttad)
        metrics_test["computation_time"].append(end_time_mincuttad - start_time_mincuttad)

        if parameters["task_type"] == "supervised":
            homogeneity_score, completeness_score, v_measure_score = metrics_calculation_supervised(
                labels_all_chromosomes, y_all_chromosomes)
            run["homogeneity_score_test"].log(homogeneity_score)
            run["completeness_score_test"].log(completeness_score)
            run["v_measure_score_test"].log(v_measure_score)
            metrics_test["homogeneity_score"].append(homogeneity_score)
            metrics_test["completeness_score"].append(completeness_score)
            metrics_test["v_measure_score"].append(v_measure_score)
            accuracy_score, precision_score, roc_auc_score, f1_score = calculate_classification_metrics(
                labels_all_chromosomes, scores_all_chromosomes, y_all_chromosomes)
            run["accuracy_test"].log(accuracy_score)
            run["precision_test"].log(precision_score)
            run["roc_auc_test"].log(roc_auc_score)
            run["f1_test"].log(f1_score)
            metrics_test["accuracy"].append(accuracy_score)
            metrics_test["precision"].append(precision_score)
            metrics_test["roc_auc"].append(roc_auc_score)
            metrics_test["f1"].append(f1_score)
        run["silhouette_score_test"].log(np.mean(silhouette_scores_chromosomes))
        run["calinski_harabasz_score_test"].log(np.mean(calinski_harabasz_scores_chromosomes))
        run["davies_bouldin_score_test"].log(np.mean(davies_bouldin_scores_chromosomes))
        metrics_test["silhouette_score"].append(np.mean(silhouette_scores_chromosomes))
        metrics_test["calinski_harabasz_score"].append(np.mean(calinski_harabasz_scores_chromosomes))
        metrics_test["davies_bouldin_score"].append(np.mean(davies_bouldin_scores_chromosomes))

    if parameters["task_type"] == "supervised":
        save_classification_metrics(parameters, scores_all_chromosomes, y_all_chromosomes)
        plot_roc_curve(parameters, [y_all_chromosomes], [scores_all_chromosomes])
        predicted_tads = determine_tad_regions(predicted_tads)

    if parameters["cross_run"]:
        save_tad_list(parameters, predicted_tads, dataloader_test, "MinCutTAD", extension = parameters["cross_run"])
    else:
        save_tad_list(parameters, predicted_tads, dataloader_test, "MinCutTAD")

    if parameters["generate_gnn_explanations"]:
        os.makedirs(os.path.join(parameters["output_directory"], parameters["dataset_name"], "node_explanations"), exist_ok=True)
        apply_gnnexplainer(parameters, model, device, dataloader_test)

    return predicted_tads

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train and evaluate a model on dataset created in preprocessing pipeline.')
    parser.add_argument("--path_parameters_json", help="path to JSON with parameters.", type=str, default='../tad_detection/model/parameters.json')
    args = parser.parse_args()
    path_parameters_json = args.path_parameters_json
    # path_parameters_json = "./tad_detection/model/parameters.json"

    parameters = load_parameters(path_parameters_json)
    os.makedirs(parameters["output_directory"], exist_ok=True)
    os.makedirs(os.path.join(parameters["output_directory"], parameters["dataset_name"]), exist_ok=True)
    os.makedirs(os.path.join(parameters["output_directory"], parameters["dataset_name"], "plots"), exist_ok=True)
    os.makedirs(os.path.join(parameters["output_directory"], parameters["dataset_name"], "metrics"), exist_ok=True)
    os.makedirs(os.path.join(parameters["output_directory"], parameters["dataset_name"], "models"), exist_ok=True)
    dump_parameters(parameters)

    logger = set_up_logger('training', parameters)
    logger.debug('Start training logger.')

    run = set_up_neptune(parameters)

    logger.info("Load data and prepare dataloaders.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, max_num_nodes, edge_index, edge_attr, y, source_information = load_data(parameters)
    parameters["max_num_nodes"] = max_num_nodes

    data_train_cross_1, data_test_cross_1, data_val_cross_1, data_train_cross_2, data_test_cross_2, data_val_cross_2 = split_data(parameters, X, edge_index, edge_attr, y, source_information)
    dataloader_train_cross_1, dataloader_test_cross_1, dataloader_val_cross_1, dataloader_train_cross_2, dataloader_test_cross_2, dataloader_val_cross_2 = torch_geometric_data_generation_dataloader(data_train_cross_1, data_test_cross_1, data_val_cross_1, data_train_cross_2, data_test_cross_2, data_val_cross_2)

    logger.info("Training starts.")
    model_for_test = train_wrapper(parameters, dataloader_train_cross_1, dataloader_val_cross_1, dataloader_train_cross_2, dataloader_val_cross_2)
    logger.info("Training ends.")

    if dataloader_val_cross_2 == 0:
        logger.info("Validation starts.")
        predicted_tads = final_validation(parameters, model_for_test, dataloader_test_cross_1, device, logger, run)
        logger.info("Validation ends.")
    else:
        logger.info("Cross validation between cell lines GM12878 and IMR-90 starts.")
        predicted_tads_cross_1, predicted_tads_cross_2 = final_validation_cross(
            parameters, model_for_test, dataloader_test_cross_1, dataloader_test_cross_2, device, logger, run)
        logger.info("Validation ends.")
