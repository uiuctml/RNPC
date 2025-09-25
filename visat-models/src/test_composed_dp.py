#!/usr/bin/env python3
import pdb

import argument
import composition
import dataset
import header
import json
import logger
import model
import os
import sklearn.metrics
import spn
import torch
import tqdm
import type
import utility
import wandb
import math
import csv

# NEW
def add_noise_to_inputs(inputs, eps=1.0, delta=0.05, sensitivity=1.0, attack_bound=0.1):
    noise_scale = math.sqrt(2 * math.log(1.25 / delta)) * sensitivity * (attack_bound / eps)
    noise = torch.normal(0, noise_scale, size=inputs.size(), device=inputs.device)
    return inputs + noise

def test(model_decomposed, spn_joint, spn_marginal, spn_settings_joint, data_loader, device, batch_step, marginal_probabilities_counted = None):
    utility.loadCheckpointBest(header.config_decomposed["dir_checkpoints"], header.config_decomposed["file_name_checkpoint_best"], model_decomposed)
    utility.loadCheckpointBestSPN(spn_joint, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint_best"])
    utility.loadCheckpointBestSPN(spn_marginal, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint_best"])
    csv_results = {}

    accuracy_epoch_composed = 0
    accuracy_epoch_list_decomposed = []
    config_dataset = data_loader.dataset.config
    ground_truths_epoch_composed = []
    ground_truths_epoch_list_decomposed = []
    mpes = {}
    output_list_composed = []
    output_list_decomposed = []
    output_list_decomposed_accuracy = []
    output_list_decomposed_precision = []
    output_list_decomposed_recall = []
    predictions_epoch_composed = []
    predictions_epoch_list_decomposed = []
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 0, leave = False)
    spn_output_rows = len(data_loader.dataset.classes_original)
    spn_output_cols = 1

    for attribute in config_dataset["attributes"]:
        attribute_labels = attribute["labels"]
        spn_output_cols *= len(attribute_labels)
        accuracy_epoch_list_decomposed.append(0)
        ground_truths_epoch_list_decomposed.append([])
        predictions_epoch_list_decomposed.append([])

    model_decomposed.eval()
    progress_bar.set_description_str("[INFO]: Testing progress")

    with torch.set_grad_enabled(False):
        for (batch_index, (input, labels_decomposed, labels_original, input_file_paths)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels_decomposed = labels_decomposed.to(device, non_blocking = True)
            labels_original = labels_original.to(device, non_blocking = True)

            # NEW: Initialize tensor to accumulate outputs
            accumulated_outputs = None

            for iteration in range(header.config_decomposed["num_noise_draw"]):
                # Set random seed for reproducibility
                torch.manual_seed(header.seed + iteration)

                # Add noise to the inputs
                noisy_input = add_noise_to_inputs(
                    input,
                    eps=header.config_decomposed["dp_eps"],
                    delta=header.config_decomposed["dp_delta"],
                    sensitivity=header.config_decomposed["dp_sensitivity"],
                    attack_bound=header.config_decomposed["attack_bound"],
                )

                # Get outputs from the model
                outputs_iteration, _ = model_decomposed(noisy_input)

                # Accumulate outputs
                if accumulated_outputs is None:
                    accumulated_outputs = outputs_iteration
                else:
                    accumulated_outputs = [accumulated_outputs[tmp] + outputs_iteration[tmp] for tmp in range(len(outputs_iteration))]

            # Average the outputs over all iterations
            outputs_decomposed = [outputs_attribute / header.config_decomposed["num_noise_draw"] for outputs_attribute in accumulated_outputs]
            # (outputs_decomposed, _) = model_decomposed(input)

            outputs_decomposed = utility.applySoftmaxDecomposed(outputs_decomposed)

            # # NEW: permute all rows such that index n is moved to index n+1
            # perm_id = 2
            # perm = torch.arange(outputs_decomposed[perm_id].shape[1]) + 1
            # perm[-1] = 0
            # outputs_decomposed[perm_id] = outputs_decomposed[perm_id][:, perm]

            for (i, dataset_entry) in enumerate(config_dataset["attributes"]):
                (_, predictions_decomposed) = torch.max(outputs_decomposed[i], 1)

                corrects_decomposed = torch.sum(predictions_decomposed == labels_decomposed[:, i].data).item()
                accuracy_batch_decomposed = corrects_decomposed / input.size(0)
                accuracy_epoch_list_decomposed[i] += corrects_decomposed

                wandb.log({"testing/batch/" + dataset_entry["name"] + "/accuracy": accuracy_batch_decomposed})

                ground_truths_epoch_list_decomposed[i] += labels_decomposed[:, i].data.tolist()
                predictions_epoch_list_decomposed[i] += predictions_decomposed.tolist()

            (matrix_a, matrix_b, outputs_composed) = composition.Composition.spn(outputs_decomposed, spn_joint, spn_marginal, spn_output_rows, spn_output_cols, device, marginal_probabilities_counted)
            (_, predictions_composed) = torch.max(outputs_composed, 1)

            corrects_composed = torch.sum(predictions_composed == labels_original.data).item()

            accuracy_batch_composed = corrects_composed / input.size(0)
            accuracy_epoch_composed += corrects_composed

            mpe_attributes = utility.findMPEs(matrix_a, matrix_b, spn_settings_joint, labels_original)
            utility.saveMPEs(input_file_paths, mpes, mpe_attributes, data_loader.dataset, labels_decomposed, labels_original, outputs_decomposed, outputs_composed)

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            wandb.log({"testing/batch/accuracy": accuracy_batch_composed})
            wandb.log({"testing/batch/step": batch_step})

            ground_truths_epoch_composed += labels_original.data.tolist()
            predictions_epoch_composed += predictions_composed.tolist()

            batch_step += 1

    progress_bar.close()

    if not os.path.isdir(header.dir_output_mpe):
        os.makedirs(header.dir_output_mpe, exist_ok = True)

    with open(os.path.join(header.dir_output_mpe, header.file_name_mpe), "w") as file_mpe:
        json.dump(mpes, file_mpe, indent = 4)

    for (i, dataset_entry) in enumerate(config_dataset["attributes"]):
        accuracy_epoch_list_decomposed[i] /= len(data_loader.dataset)
        precision_epoch_decomposed = sklearn.metrics.precision_score(ground_truths_epoch_list_decomposed[i], predictions_epoch_list_decomposed[i], average = "macro", zero_division = 0)
        recall_epoch_decomposed = sklearn.metrics.recall_score(ground_truths_epoch_list_decomposed[i], predictions_epoch_list_decomposed[i], average = "macro", zero_division = 0)

        output_list_decomposed_accuracy.append(accuracy_epoch_list_decomposed[i])
        output_list_decomposed_precision.append(precision_epoch_decomposed)
        output_list_decomposed_recall.append(recall_epoch_decomposed)

        wandb.log({"testing/epoch/" + dataset_entry["name"] + "/accuracy": accuracy_epoch_list_decomposed[i]})
        wandb.log({"testing/epoch/" + dataset_entry["name"] + "/precision": precision_epoch_decomposed})
        wandb.log({"testing/epoch/" + dataset_entry["name"] + "/recall": recall_epoch_decomposed})
        wandb.summary["testing/epoch/" + dataset_entry["name"] + "/accuracy"] = accuracy_epoch_list_decomposed[i]
        wandb.summary["testing/epoch/" + dataset_entry["name"] + "/precision"] = precision_epoch_decomposed
        wandb.summary["testing/epoch/" + dataset_entry["name"] + "/recall"] = recall_epoch_decomposed

        logger.log_info("Decomposed testing accuracy for \"" + dataset_entry["name"] + "\": " + str(accuracy_epoch_list_decomposed[i]) + ".")
        logger.log_trace("Decomposed testing precision for \"" + dataset_entry["name"] + "\": " + str(precision_epoch_decomposed) + ".")
        logger.log_trace("Decomposed testing recall for \"" + dataset_entry["name"] + "\": " + str(recall_epoch_decomposed) + ".")
        csv_results[dataset_entry["name"]] = accuracy_epoch_list_decomposed[i]

    accuracy_epoch_composed /= len(data_loader.dataset)

    precision_epoch_composed = sklearn.metrics.precision_score(ground_truths_epoch_composed, predictions_epoch_composed, average = "macro", zero_division = 0)
    recall_epoch_composed = sklearn.metrics.recall_score(ground_truths_epoch_composed, predictions_epoch_composed, average = "macro", zero_division = 0)
    output_list_composed += [accuracy_epoch_composed, precision_epoch_composed, recall_epoch_composed]
    output_list_decomposed += output_list_decomposed_accuracy
    output_list_decomposed += output_list_decomposed_precision
    output_list_decomposed += output_list_decomposed_recall

    wandb.log({"testing/epoch/accuracy": accuracy_epoch_composed})
    wandb.log({"testing/epoch/precision": precision_epoch_composed})
    wandb.log({"testing/epoch/recall": recall_epoch_composed})
    wandb.summary["testing/epoch/accuracy"] = accuracy_epoch_composed
    wandb.summary["testing/epoch/precision"] = precision_epoch_composed
    wandb.summary["testing/epoch/recall"] = recall_epoch_composed

    logger.log_info("Composed testing accuracy: " + str(accuracy_epoch_composed) + ".")
    logger.log_trace("Composed testing precision: " + str(precision_epoch_composed) + ".")
    logger.log_trace("Composed testing recall: " + str(recall_epoch_composed) + ".")
    csv_results["class"] = accuracy_epoch_composed

    utility.logTestOutput(output_list_composed, header.config_baseline, config_dataset, True)
    utility.logTestOutput(output_list_decomposed, header.config_decomposed, config_dataset)

    # Write csv results
    fieldnames = csv_results.keys()
    csv_dir = os.path.join(header.config_decomposed["dir_results"], "test_composed_dp")
    os.makedirs(csv_dir, exist_ok=True)
    csv_file = os.path.join(csv_dir, "noattack.csv")
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(csv_results)

    return

def main():
    argument.processArgumentsTestComposed()

    header.run_name_baseline = header.config_decomposed["run_name"]
    header.config_baseline["dir_dataset_test"] = header.config_decomposed["dir_dataset_test"]
    header.config_baseline["file_name_checkpoint"] = header.run_name_decomposed + ".tar"
    header.config_baseline["file_name_checkpoint_best"] = header.run_name_decomposed + ".best.tar"
    header.config_baseline["run_name"] = header.run_name_decomposed

    utility.setSeed(header.seed)
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    config = {
        "decomposed": header.config_decomposed,
        "spn": header.config_spn
    }

    wandb.init(config = config, mode = "disabled")

    dataset_transforms = utility.createTransform(header.config_decomposed)
    dataset_test = dataset.VISATDataset(header.config_decomposed["dir_dataset_test"], dataset_transforms)
    config_dataset = dataset_test.config
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_decomposed["data_loader_batch_size"], shuffle = False, num_workers = header.config_decomposed["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    marginal_probabilities_counted = None
    model_decomposed = model.createModelDecomposed(device)
    model_decomposed = torch.nn.DataParallel(model_decomposed)
    model_decomposed = model_decomposed.to(device)
    spn_joint = spn.SPN(device)
    spn_marginal = spn.SPN(device)

    if header.config_spn["optimizer"] == type.OptimizerSPN.cccp_discriminative.name:
        (marginal_probabilities_counted, _) = utility.countAttributeJointProbabilities(config_dataset, device)

    logger.log_info("Loading SPN from \"" + header.config_spn["file_path_spn"] + "\"...")

    spn_joint.load(header.config_spn["file_path_spn"])
    spn_marginal.load(header.config_spn["file_path_spn"])

    logger.log_info("Loading SPN leaf node settings...")

    spn_settings_joint = utility.generateSPNSettings(config_dataset, device)
    spn_settings_marginal = torch.clone(spn_settings_joint)
    spn_settings_marginal[:, -1] = -1

    logger.log_info("Setting SPN leaf nodes...")

    spn_joint.set_leaf_nodes(spn_settings_joint)
    spn_marginal.set_leaf_nodes(spn_settings_marginal)

    if header.show_model_summary:
        logger.log_info("Number of nodes: " + str(len(spn_joint.nodes)) + ".")
        logger.log_info("Number of sum nodes: " + str(len(spn_joint.sum_nodes)) + ".")
        logger.log_info("Number of product nodes: " + str(len(spn_joint.product_nodes)) + ".")
        logger.log_info("Number of leaf nodes: " + str(len(spn_joint.leaf_nodes)) + ".")
        logger.log_info("SPN depths: " + str(spn_joint.depth) + ".")
        logger.log_info("SPN leaf node setting dimension: (" + str(int(spn_settings_joint.shape[0])) + ", " + str(int(spn_settings_joint.shape[1])) + ").")

    test(model_decomposed, spn_joint, spn_marginal, spn_settings_joint, data_loader_test, device, 1, marginal_probabilities_counted)

    return

if __name__ == "__main__":
    main()
