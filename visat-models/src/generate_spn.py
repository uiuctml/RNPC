#!/usr/bin/env python3

import header
import json
import logger
import os
import torch
import utility

def main():
    device = torch.device("cuda")

    file_config_dataset = open(header.file_path_dataset_config, "r")
    config_dataset = json.load(file_config_dataset)
    file_config_dataset.close()

    # Get visual attribute labeling data
    labels_attribute = utility.getLabelsAttribute(config_dataset)
    labels_original = utility.getLabelsOriginal(config_dataset)
    labels_attribute_indices = utility.getIndicesFromLabelsAttribute(labels_attribute)
    labels_original_indices = utility.getIndicesFromLabelsOriginal(labels_original)
    (_, label_probabilities) = utility.countAttributeJointProbabilities(config_dataset, device)

    attribute_index_original_label = len(config_dataset["attributes"])
    cat_node_dict = {}
    edge_count_sum_prd = 0
    edge_count_prd_leaf = 0
    lines_edges = "##EDGES##\n"
    lines_nodes = "##NODES##\n"
    node_count_leaf = 0
    node_count_prd = 0
    node_count_sum = 0
    node_sequence = 0
    node_sequence_root = node_sequence

    # Add sum root node
    lines_nodes += str(node_sequence_root) + ",SUM\n"
    node_sequence += 1
    node_count_sum += 1

    # Log visual attribute statistics
    for attribute_name in labels_attribute.keys():
        logger.log_info("Number of categories for attribute \"" + attribute_name + "\": " + str(len(labels_attribute[attribute_name])) + ".")

    # Add visual attribute leaf nodes
    for (attribute_index, attribute) in enumerate(config_dataset["attributes"]):
        for category_index in labels_attribute_indices[attribute["name"]].values():
            line_cat_node = "CATNODEPRD," + str(attribute_index) + "," + str(category_index)
            cat_node_dict[line_cat_node] = node_sequence
            lines_nodes += str(node_sequence) + "," + line_cat_node + "\n"
            node_sequence += 1
            node_count_leaf += 1

    # Add original label leaf nodes
    for category_index in range(0, len(config_dataset["mappings"])):
        line_cat_node = "CATNODEPRD," + str(attribute_index_original_label) + "," + str(category_index)
        cat_node_dict[line_cat_node] = node_sequence
        lines_nodes += str(node_sequence) + "," + line_cat_node + "\n"
        node_sequence += 1
        node_count_leaf += 1

    for label_original in label_probabilities.keys():
        category_indices = label_probabilities[label_original][0]
        frequency = label_probabilities[label_original][1]

        # Add product nodes
        node_sequence_prd = node_sequence
        lines_nodes += str(node_sequence_prd) + ",PRD\n"
        node_sequence += 1
        node_count_prd += 1

        # Add root-to-product edges
        lines_edges += str(node_sequence_root) + "," + str(node_sequence_prd) + "," + str(frequency) + "\n"
        edge_count_sum_prd += 1

        # Add product-to-visual-attribute-leaf edges
        for (attribute_index, category_index) in enumerate(category_indices):
            line_cat_node = "CATNODEPRD," + str(attribute_index) + "," + str(category_index)
            node_sequence_cat = cat_node_dict[line_cat_node]
            lines_edges += str(node_sequence_prd) + "," + str(node_sequence_cat) + "\n"
            edge_count_prd_leaf += 1

        # Add product-to-orignal-label-leaf edges
        category_index_original_label = labels_original_indices[label_original]
        line_cat_node = "CATNODEPRD," + str(attribute_index_original_label) + "," + str(category_index_original_label)
        node_sequence_cat = cat_node_dict[line_cat_node]
        lines_edges += str(node_sequence_prd) + "," + str(node_sequence_cat) + "\n"
        edge_count_prd_leaf += 1

    logger.log_info("Number of sum nodes: " + str(node_count_sum) + ".")
    logger.log_info("Number of product nodes: " + str(node_count_prd) + ".")
    logger.log_info("Number of leaf nodes: " + str(node_count_leaf) + ".")
    logger.log_info("Number of sum-to-product edges: " + str(edge_count_sum_prd) + ".")
    logger.log_info("Number of product-to-leaf edges: " + str(edge_count_prd_leaf) + ".")

    lines = lines_nodes + lines_edges

    if not os.path.isdir(header.dir_output_spn):
        os.makedirs(header.dir_output_spn, exist_ok = True)

    file_path_spn_manual = os.path.join(header.dir_output_spn, header.file_name_spn_manual)

    with open(file_path_spn_manual, "w") as file_spn_manual:
        file_spn_manual.writelines(lines)

    logger.log_info("Wrote to \"" + file_path_spn_manual + "\".")

    return

if __name__ == "__main__":
    main()
