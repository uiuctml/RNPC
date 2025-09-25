#!/usr/bin/env python3

import argument
import header
import json
import logger
import spn
import torch
import utility

def generateSPNSettings(config_dataset, device):
    labels_attribute = utility.getLabelsAttribute(config_dataset)
    labels_original = utility.getLabelsOriginal(config_dataset)
    indices_attribute = utility.getIndicesFromLabelsAttribute(labels_attribute)
    indices_original = utility.getIndicesFromLabelsOriginal(labels_original)
    spn_settings = []

    for label_original in config_dataset["mappings"].keys():
        spn_setting = []

        for attribute_name in config_dataset["mappings"][label_original]["labels"].keys():
            label_attribute = config_dataset["mappings"][label_original]["labels"][attribute_name]
            index_attribute = indices_attribute[attribute_name][label_attribute]
            spn_setting.append(index_attribute)

        index_original = indices_original[label_original]
        spn_setting.append(index_original)
        spn_settings.append(spn_setting)

    spn_settings = torch.Tensor(spn_settings)
    spn_settings = spn_settings.to(device)

    return spn_settings

def main():
    argument.processArgumentsTestSPN()

    utility.setSeed(header.seed)
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    file_config_dataset = open(header.file_path_dataset_config, "r")
    config_dataset = json.load(file_config_dataset)
    file_config_dataset.close()

    attribute_sizes = []
    device = torch.device("cuda")
    spn_joint = spn.SPN()
    spn_marginal = spn.SPN()
    spn_output_rows = len(config_dataset["mappings"])
    spn_output_cols = 1

    for attribute in config_dataset["attributes"]:
        attribute["labels"].remove("")
        attribute_labels = attribute["labels"]
        spn_output_cols *= len(attribute_labels)
        attribute_sizes.append(len(attribute_labels))

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

    utility.loadCheckpointBestSPN(spn_joint, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint_best"])
    utility.loadCheckpointBestSPN(spn_marginal, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint_best"])

    if header.show_model_summary:
        logger.log_info("Number of nodes: " + str(len(spn_joint.nodes)) + ".")
        logger.log_info("Number of sum nodes: " + str(len(spn_joint.sum_nodes)) + ".")
        logger.log_info("Number of product nodes: " + str(len(spn_joint.product_nodes)) + ".")
        logger.log_info("Number of leaf nodes: " + str(len(spn_joint.leaf_nodes)) + ".")
        logger.log_info("SPN depths: " + str(spn_joint.depth) + ".")
        logger.log_info("SPN leaf node setting dimension: (" + str(int(spn_settings_joint.shape[0])) + ", " + str(int(spn_settings_joint.shape[1])) + ").")

    log_likelihoods_joint = spn_joint.forward()
    log_likelihoods_marginal = spn_marginal.forward()

    logger.log_trace("Testing joint log likelihoods: " + str(log_likelihoods_joint) + ".")
    logger.log_trace("Testing marginal log likelihoods: " + str(log_likelihoods_marginal) + ".")

    (_, label_probabilities) = utility.countAttributeJointProbabilities(config_dataset, device)
    matrix_a_joint_spn = torch.exp(log_likelihoods_joint.reshape(spn_output_rows, spn_output_cols))
    matrix_a_joint_counted = torch.zeros(matrix_a_joint_spn.shape).to(device)
    (_, attribute_indices_to_matrix_a_col_indices) = utility.getMatrixAColIndicesAttributeIndicesMaps(spn_output_cols, spn_settings_joint)

    for (index_row, label_original) in enumerate(label_probabilities.keys()):
        indices_attribute = label_probabilities[label_original][0]
        joint_probability_counted = label_probabilities[label_original][1]
        index_col = attribute_indices_to_matrix_a_col_indices[indices_attribute]
        matrix_a_joint_counted[index_row][index_col] = joint_probability_counted

    error = torch.sum(torch.abs(matrix_a_joint_spn - matrix_a_joint_counted)).item() / 2
    accuracy = 1 - error

    logger.log_info("Testing accuracy: " + str(accuracy) + ".")

    return

if __name__=="__main__":
    main()
