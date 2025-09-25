import pdb

import cv2
import header
import itertools
import logger
import natsort
import numpy
import os
import random
import torch
import torchvision
import wandb

def applySoftmax(output):
    return torch.nn.functional.softmax(output, dim = 1)

def applySoftmaxDecomposed(outputs_decomposed):
    outputs_decomposed_softmax = []

    for i in range(len(outputs_decomposed)):
        outputs_decomposed_softmax.append(applySoftmax(outputs_decomposed[i]))

    return outputs_decomposed_softmax

def computeCovarianceRegularization(features):
    loss = 0
    features_centered = []

    # Mean-center all attribute feature tensors across rows
    for feature in features:
        means_col = torch.mean(feature, axis = 0)
        features_centered.append(feature.detach().clone() - means_col)

    # Obtain unique pairs of attribute feature tensors
    feature_pairs = list(itertools.combinations(features_centered, 2))

    for feature_pair in feature_pairs:
        # Compute pair-wise covariance matrix
        convariance = torch.matmul(feature_pair[0].t(), feature_pair[1])

        # Add Frobenius norm to total loss
        loss += torch.sum(torch.square(convariance))

    # Multiply total loss with tunable regularization factor
    loss *= header.config_decomposed["factor_loss_covariance"]

    return loss

def computeL2Norm(parameters):
    parameters_list = []

    for parameter in parameters:
        parameters_list.append(parameter.view(-1))

    return torch.square(torch.cat(parameters_list)).sum().item()

def countAttributeJointProbabilities(config_dataset, device):
    attribute_ranges = []
    label_probabilities = {}
    labels_attribute = getLabelsAttribute(config_dataset)
    labels_attribute_indices = getIndicesFromLabelsAttribute(labels_attribute)
    dataset_size = 0

    for attribute in labels_attribute.keys():
        attribute_count = len(labels_attribute[attribute])
        attribute_range = torch.Tensor(range(attribute_count))
        attribute_ranges.append(attribute_range)

    attributes_indices = torch.cartesian_prod(*attribute_ranges)
    joint_probabilities = torch.zeros(attributes_indices.shape[0], 1)
    joint_probabilities_indices = {}

    for (i, attribute_indices) in enumerate(attributes_indices.tolist()):
        joint_probabilities_indices[tuple(attribute_indices)] = i

    for (class_label, class_attributes) in config_dataset["mappings"].items():
        attribute_indices = []
        # Convert attribute strings to corresponding tensor indices
        for (attribute_name, attribute_label) in class_attributes["labels"].items():
            attribute_indices.append(labels_attribute_indices[attribute_name][attribute_label])

        attribute_indices = tuple(attribute_indices)
        class_size = len(os.listdir(os.path.join(header.config_decomposed["dir_dataset_train"], class_label)))
        label_probabilities[class_label] = (attribute_indices, class_size)
        dataset_size += class_size

    for (class_label, _) in config_dataset["mappings"].items():
        label_probabilities[class_label] = (label_probabilities[class_label][0], label_probabilities[class_label][1] / dataset_size)

    for (attribute_indices, probabilities) in label_probabilities.values():
        index = joint_probabilities_indices[attribute_indices]
        joint_probabilities[index] = probabilities

    joint_probabilities = joint_probabilities.repeat(len(config_dataset["mappings"]), 1)
    joint_probabilities = joint_probabilities.squeeze()

    return (joint_probabilities.to(device), label_probabilities)

def createTransform(config):
    dataset_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((config["model_input_height"], config["model_input_width"])),
        torchvision.transforms.ToTensor()
    ])

    if config["input_grayscale"]:
        dataset_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((config["model_input_height"], config["model_input_width"])),
        torchvision.transforms.Grayscale(3),
        torchvision.transforms.ToTensor()
    ])

    return dataset_transforms

def findMPEs(matrix_a, matrix_b, spn_settings, labels_original):
    (matrix_a_col_indices_to_attribute_indices, _) = getMatrixAColIndicesAttributeIndicesMaps(matrix_a.shape[1], spn_settings)
    mpe_attributes = []

    matrix_a = matrix_a.t() # product of category size of all attributes x number of original labels
    matrix_a = torch.index_select(matrix_a, 1, labels_original)   # product of category size of all attributes x batch size
    matrix_c = matrix_a * matrix_b  # product of category size of all attributes x batch size
    mpes_matrix_a_col_indices = torch.argmax(matrix_c, 0)   # 0 x batch size

    for mpes_matrix_a_col_index in mpes_matrix_a_col_indices.cpu().tolist():
        mpe_attributes.append(matrix_a_col_indices_to_attribute_indices[mpes_matrix_a_col_index])

    return mpe_attributes

# def getBinaryLabelsDecomposed(labels_decomposed):
#     for i in range(len(labels_decomposed)):
#         labels_decomposed[i] = (labels_decomposed[i] > 0).float()
#
#     return torch.cat(labels_decomposed, dim = 1)
def getBinaryLabelsDecomposed(labels_decomposed, data_loader):
    batch_size, num_attributes = labels_decomposed.shape
    labels_attribute = data_loader.dataset.classes
    total_binary_columns = sum(len(values) for values in labels_attribute.values())

    # Initialize the output binary tensor
    binary_labels_decomposed = torch.zeros((batch_size, total_binary_columns), dtype=torch.float32)

    # Convert labels to binary columns
    start_idx = 0
    for col_idx, (attribute, values) in enumerate(labels_attribute.items()):
        num_values = len(values)
        for value_idx in range(num_values):
            # Assign 1 where the value matches, otherwise 0
            binary_labels_decomposed[:, start_idx + value_idx] = (labels_decomposed[:, col_idx] == value_idx).float()
        start_idx += num_values  # Move start index for the next attribute

    return binary_labels_decomposed

def getBinaryLabelsOriginal(labels_original, data_loader):
    return torch.nn.functional.one_hot(labels_original, num_classes = len(data_loader.dataset.classes_original)).float()

def getLabelsAttribute(dataset_config):
    labels_attribute = {}

    for attribute in dataset_config["attributes"]:
        if "" in attribute["labels"]:
            attribute["labels"].remove("")

        labels_attribute[attribute["name"]] = attribute["labels"]

    return labels_attribute

def getLabelsOriginal(dataset_config):
    if "instance_wise" in dataset_config and dataset_config["instance_wise"]:
        labels_original = []
        labels_original_set = set()

        for image_name in dataset_config["mappings"].keys():
            class_name = image_name.split('/')[0]

            if class_name not in labels_original_set:
                labels_original.append(class_name)
                labels_original_set.add(class_name)

        return natsort.natsorted(labels_original)
    else:
        return list(dataset_config["mappings"].keys())

def getIndicesFromLabelsAttribute(labels_attribute):
    indices = {}

    for attribute in labels_attribute.keys():
        labels_to_indices = {}

        for i in range(len(labels_attribute[attribute])):
            labels_to_indices[labels_attribute[attribute][i]] = i

        indices[attribute] = labels_to_indices

    return indices

def getIndicesFromLabelsOriginal(labels_original):
    labels_to_indices = {}

    for i in range(len(labels_original)):
        labels_to_indices[labels_original[i]] = i

    return labels_to_indices

def getMatrixAColIndicesAttributeIndicesMaps(matrix_a_cols, spn_settings):
    attribute_indices_list = spn_settings[:matrix_a_cols, :-1].cpu().int().tolist()
    matrix_a_col_indices_to_attribute_indices = {}
    attribute_indices_to_matrix_a_col_indices = {}

    for (matrix_a_row_index, attribute_indices) in enumerate(attribute_indices_list):
        matrix_a_col_indices_to_attribute_indices[matrix_a_row_index] = tuple(attribute_indices)
        attribute_indices_to_matrix_a_col_indices[tuple(attribute_indices)] = matrix_a_row_index

    return (matrix_a_col_indices_to_attribute_indices, attribute_indices_to_matrix_a_col_indices)

def generateSPNSettings(config_dataset, device):
    attribute_ranges = []
    labels_attribute = getLabelsAttribute(config_dataset)
    labels_original = getLabelsOriginal(config_dataset)

    for attribute in labels_attribute.keys():
        attribute_count = len(labels_attribute[attribute])
        attribute_range = torch.Tensor(range(attribute_count))
        attribute_range = attribute_range.to(device)
        attribute_ranges.append(attribute_range)

        logger.log_trace("Number of attribute labels for \"" + attribute + "\": " + str(attribute_count) + ".")

    original_count = len(labels_original)
    original_range = torch.Tensor(range(original_count))
    original_range = original_range.to(device)

    logger.log_trace("Number of original labels: " + str(original_count) + ".")

    spn_settings = torch.cartesian_prod(*attribute_ranges)

    original_range = original_range.repeat_interleave(spn_settings.shape[0]).reshape(-1, 1)
    spn_settings = spn_settings.repeat(original_count, 1)
    spn_settings = torch.cat((spn_settings, original_range), 1)

    return spn_settings

def loadCheckpoint(dir_checkpoints, file_name_checkpoint, accuracy_validation_best, batch_step_train, batch_step_validate, criterions, epoch, learning_rate_schedulers, model, optimizers):
    if wandb.run.resumed:
        if not os.path.isdir(dir_checkpoints):
            os.makedirs(dir_checkpoints, exist_ok = True)

        try:
            wandb.restore(file_name_checkpoint, root = dir_checkpoints)
        except:
            pass
        else:
            logger.log_info("Restored checkpoint \"" + file_name_checkpoint + "\" from Weights & Biases.")

        file_path_checkpoint = os.path.join(dir_checkpoints, file_name_checkpoint)

        if os.path.isfile(file_path_checkpoint):
            checkpoint = torch.load(file_path_checkpoint)
            accuracy_validation_best = checkpoint["accuracy_validation_best"]
            batch_step_train = checkpoint["batch_step_train"]
            batch_step_validate = checkpoint["batch_step_validate"]
            criterions = checkpoint["criterions"]
            epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["model_state_dict"])

            for (learning_rate_scheduler, learning_rate_scheduler_state_dict) in zip(learning_rate_schedulers, checkpoint["learning_rate_scheduler_state_dict_list"]):
                learning_rate_scheduler.load_state_dict(learning_rate_scheduler_state_dict)

            for (optimizer, optimizer_state_dict) in zip(optimizers, checkpoint["optimizer_state_dict_list"]):
                optimizer.load_state_dict(optimizer_state_dict)

            logger.log_info("Loaded checkpoint \"" + file_name_checkpoint + "\".")
        else:
            logger.log_fatal("Checkpoint file \"" + file_name_checkpoint + "\" missing.")
            exit(-1)

    return (accuracy_validation_best, batch_step_train, batch_step_validate, criterions, epoch)

def loadCheckpointBest(dir_checkpoints, file_name_checkpoint, model):
    if not os.path.isdir(dir_checkpoints):
        os.makedirs(dir_checkpoints, exist_ok = True)

    try:
        wandb.restore(file_name_checkpoint, root = dir_checkpoints)
    except:
        pass
    else:
        logger.log_info("Restored checkpoint \"" + file_name_checkpoint + "\" from Weights & Biases.")

    file_path_checkpoint = os.path.join(dir_checkpoints, file_name_checkpoint)

    if os.path.isfile(file_path_checkpoint):
        checkpoint = torch.load(file_path_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])

        logger.log_info("Loaded checkpoint \"" + file_name_checkpoint + "\".")
    else:
        logger.log_fatal("Checkpoint file \"" + file_name_checkpoint + "\" missing.")
        exit(-1)

    return

def loadCheckpointBestSPN(spn, dir_checkpoints, file_name_checkpoint):
    if not os.path.isdir(dir_checkpoints):
        logger.log_fatal("Checkpoint directory \"" + dir_checkpoints + "\" missing.")
        exit(-1)

    file_path_checkpoint = os.path.join(dir_checkpoints, file_name_checkpoint)

    if os.path.isfile(file_path_checkpoint):
        checkpoint = torch.load(file_path_checkpoint)
        spn.set_weights(checkpoint["weights"])

        logger.log_info("Loaded checkpoint \"" + file_name_checkpoint + "\".")
    else:
        logger.log_fatal("Checkpoint file \"" + file_name_checkpoint + "\" missing.")
        exit(-1)

    return

def loadCheckpointSPN(spn, dir_checkpoints, file_name_checkpoint, log_likelihood_best, log_likelihood_train_last, epoch):
    if wandb.run.resumed:
        if not os.path.isdir(dir_checkpoints):
            logger.log_fatal("Checkpoint directory \"" + dir_checkpoints + "\" missing.")
            exit(-1)

        try:
            wandb.restore(file_name_checkpoint, root = dir_checkpoints)
        except:
            pass
        else:
            logger.log_info("Restored checkpoint \"" + file_name_checkpoint + "\" from Weights & Biases.")

        file_path_checkpoint = os.path.join(dir_checkpoints, file_name_checkpoint)

        if os.path.isfile(file_path_checkpoint):
            checkpoint = torch.load(file_path_checkpoint)
            epoch = checkpoint["epoch"]
            log_likelihood_best = checkpoint["log_likelihood_best"]
            log_likelihood_train_last = checkpoint["log_likelihood_train_last"]
            spn.set_weights(checkpoint["weights"])

            logger.log_info("Loaded checkpoint \"" + file_name_checkpoint + "\".")
        else:
            logger.log_fatal("Checkpoint file \"" + file_name_checkpoint + "\" missing.")
            exit(-1)

    return (log_likelihood_best, log_likelihood_train_last, epoch)

def logTestOutput(output, config, config_dataset, composed = False):
    if config["log_test_output"] == False:
        return

    dir_dataset_test = config["dir_dataset_test"]
    dir_test_output = config["dir_test_output"]
    file_empty = False
    file_line = ""
    file_path_test_output = ""

    if dir_dataset_test[-1] == '/':
        dir_dataset_test = dir_dataset_test[:-1]

    if not os.path.isdir(dir_test_output):
        os.makedirs(dir_test_output, exist_ok = True)

    if dir_dataset_test.split('/')[-1] == "test":
        file_path_test_output = os.path.join(dir_test_output, config["file_name_test_output_clean"])
    else:
        file_path_test_output = os.path.join(dir_test_output, config["file_name_test_output_attacked"])

    if not os.path.isfile(file_path_test_output):
        file_empty = True
    elif os.stat(file_path_test_output).st_size == 0:
        file_empty = True

    file_line += str(config["seed"])
    file_line += "\t"
    file_line += config["run_name"]
    file_line += "\t"
    file_line += dir_dataset_test.split('/')[-1]
    file_line += "\t"
    file_line += str(composed)

    for value in output:
        file_line += "\t"
        file_line += str(value)

    file_line += "\n"

    with open(file_path_test_output, "a") as file_test_output:
        if file_empty:
            if config["type"] == "baseline":
                file_test_output.write("# seed\tmodel\tattack\tcomposed\taccuracy\tprecision\trecall\n")
            elif config["type"] == "decomposed":
                file_line_header = "# seed\tmodel\tattack\tcomposed"

                for data_type in ["accuracy", "precision", "recall"]:
                    for attribute in config_dataset["attributes"]:
                        attribute_name = attribute["name"]
                        file_line_header += "\t"
                        file_line_header += data_type + "_" + attribute_name

                file_line_header += "\n"
                file_test_output.write(file_line_header)
            else:
                logger.log_fatal("Unknown configuration type")
                exit(-1)

        file_test_output.write(file_line)

    logger.log_info("Logged test output to \"" + file_path_test_output + "\".")

    return

def lossNegativeLogLikelihood(output, label):
    label = label.reshape(-1, 1)
    output = torch.gather(output, 1, label)
    negative_log_likelihood = -1 * torch.log(output)

    return negative_log_likelihood.mean()

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    width_image = image.shape[1]
    width_resize = image.shape[1]
    height_image = image.shape[0]
    height_resize = image.shape[0]

    if width is None and height is None:
        return image

    if width is None:
        resize_ratio = height / height_image
        width_resize = int(width_image * resize_ratio)
        height_resize = height
    else:
        resize_ratio = width / width_image
        width_resize = width
        height_resize = int(height_image * resize_ratio)

    return cv2.resize(image, (width_resize, height_resize), interpolation=inter)

def saveCheckpoint(dir_checkpoints, file_name_checkpoint, accuracy_validation_best, batch_step_train, batch_step_validate, criterions, epoch, learning_rate_schedulers, model, optimizers):
    if not os.path.isdir(dir_checkpoints):
        os.makedirs(dir_checkpoints, exist_ok = True)

    learning_rate_scheduler_state_dict_list = []
    optimizer_state_dict_list = []

    for learning_rate_scheduler in learning_rate_schedulers:
        learning_rate_scheduler_state_dict_list.append(learning_rate_scheduler.state_dict())

    for optimizer in optimizers:
        optimizer_state_dict_list.append(optimizer.state_dict())

    checkpoint = {
        "accuracy_validation_best": accuracy_validation_best,
        "batch_step_train": batch_step_train,
        "batch_step_validate": batch_step_validate,
        "criterions": criterions,
        "epoch": epoch,
        "learning_rate_scheduler_state_dict_list": learning_rate_scheduler_state_dict_list,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict_list": optimizer_state_dict_list
    }
    file_path_checkpoint = os.path.join(dir_checkpoints, file_name_checkpoint)

    torch.save(checkpoint, file_path_checkpoint)

    try:
        wandb.save(file_path_checkpoint, base_path = dir_checkpoints)
    except:
        pass
    else:
        logger.log_info("Saved checkpoint \"" + file_name_checkpoint + "\" to Weights & Biases.")

    logger.log_info("Saved checkpoint \"" + file_name_checkpoint + "\".")

    return

def saveCheckpointSPN(spn, dir_checkpoints, file_name_checkpoint, log_likelihood_best, log_likelihood_train_last, epoch):
    weights = spn.get_weights()

    if not os.path.isdir(dir_checkpoints):
        os.makedirs(dir_checkpoints, exist_ok = True)

    checkpoint = {
        "epoch": epoch,
        "log_likelihood_best": log_likelihood_best,
        "log_likelihood_train_last": log_likelihood_train_last,
        "weights": weights
    }
    file_path_checkpoint = os.path.join(dir_checkpoints, file_name_checkpoint)

    torch.save(checkpoint, file_path_checkpoint)

    try:
        wandb.save(file_path_checkpoint, base_path = dir_checkpoints)
    except:
        pass
    else:
        logger.log_info("Saved checkpoint \"" + file_name_checkpoint + "\" to Weights & Biases.")

    logger.log_info("Saved checkpoint \"" + file_name_checkpoint + "\".")

    return

def saveCounterfactuals(input_file_paths, counterfactuals, dataset, labels_decomposed, labels_original, outputs_decomposed, outputs_decomposed_counterfactual, outputs_composed, outputs_composed_counterfactual):
    batch_size = labels_original.nelement()
    config_dataset = dataset.config

    for batch_index in range(batch_size):
        input_file_path = os.path.basename(input_file_paths[batch_index])

        counterfactuals[input_file_path] = {}
        counterfactuals[input_file_path]["counterfactual"] = {}
        counterfactuals[input_file_path]["ground_truth"] = {}
        counterfactuals[input_file_path]["prediction"] = {}

        for (attribute_index, attribute) in enumerate(config_dataset["attributes"]):
            attribute_name = attribute["name"]

            (outputs_decomposed_counterfactual_mpe_probability, outputs_decomposed_counterfactual_mpe_label) = torch.max(outputs_decomposed_counterfactual[attribute_index][batch_index], 0)
            (outputs_decomposed_mpe_probability, outputs_decomposed_mpe_label) = torch.max(outputs_decomposed[attribute_index][batch_index], 0)

            counterfactuals[input_file_path]["counterfactual"][attribute_name] = (dataset.classes[attribute_name][outputs_decomposed_counterfactual_mpe_label], outputs_decomposed_counterfactual_mpe_probability.item())
            counterfactuals[input_file_path]["ground_truth"][attribute_name] = dataset.classes[attribute_name][labels_decomposed[batch_index][attribute_index]]
            counterfactuals[input_file_path]["prediction"][attribute_name] = (dataset.classes[attribute_name][outputs_decomposed_mpe_label], outputs_decomposed_mpe_probability.item())

        (outputs_composed_counterfactual_mpe_probability, outputs_composed_counterfactual_mpe_label) = torch.max(outputs_composed_counterfactual[batch_index], 0)
        (outputs_composed_mpe_probability, outputs_composed_mpe_label) = torch.max(outputs_composed[batch_index], 0)

        counterfactuals[input_file_path]["counterfactual"]["original"] = (dataset.classes_original[outputs_composed_counterfactual_mpe_label], outputs_composed_counterfactual_mpe_probability.item())
        counterfactuals[input_file_path]["ground_truth"]["original"] = dataset.classes_original[labels_original[batch_index]]
        counterfactuals[input_file_path]["prediction"]["original"] = (dataset.classes_original[outputs_composed_mpe_label], outputs_composed_mpe_probability.item())

    return

def saveMPEs(input_file_paths, mpes, mpe_attributes, dataset, labels_decomposed, labels_original, outputs_decomposed, outputs_composed):
    batch_size = labels_original.nelement()
    config_dataset = dataset.config

    for batch_index in range(batch_size):
        input_file_path = os.path.basename(input_file_paths[batch_index])

        mpes[input_file_path] = {}
        mpes[input_file_path]["mpe"] = {}
        mpes[input_file_path]["ground_truth"] = {}
        mpes[input_file_path]["prediction"] = {}

        for (attribute_index, attribute) in enumerate(config_dataset["attributes"]):
            attribute_name = attribute["name"]
            (outputs_decomposed_mpe_probability, outputs_decomposed_mpe_label) = torch.max(outputs_decomposed[attribute_index][batch_index], 0)

            mpes[input_file_path]["mpe"][attribute_name] = dataset.classes[attribute_name][mpe_attributes[batch_index][attribute_index]]
            mpes[input_file_path]["ground_truth"][attribute_name] = dataset.classes[attribute_name][labels_decomposed[batch_index][attribute_index]]
            mpes[input_file_path]["prediction"][attribute_name] = (dataset.classes[attribute_name][outputs_decomposed_mpe_label], outputs_decomposed_mpe_probability.item())

        (outputs_composed_mpe_probability, outputs_composed_mpe_label) = torch.max(outputs_composed[batch_index], 0)

        mpes[input_file_path]["mpe"]["original"] = "N/A"
        mpes[input_file_path]["ground_truth"]["original"] = dataset.classes_original[labels_original[batch_index]]
        mpes[input_file_path]["prediction"]["original"] = (dataset.classes_original[outputs_composed_mpe_label], outputs_composed_mpe_probability.item())

    return

def setSeed(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    return

def wAndBDefineMetrics():
    wandb.define_metric("testing/batch/step")
    wandb.define_metric("testing/epoch/step")
    wandb.define_metric("training/batch/step")
    wandb.define_metric("training/epoch/step")
    wandb.define_metric("validation/batch/step")
    wandb.define_metric("validation/epoch/step")

    wandb.define_metric("testing/batch/*", step_metric = "testing/batch/step")
    wandb.define_metric("testing/epoch/*", step_metric = "testing/epoch/step")
    wandb.define_metric("training/batch/*", step_metric = "training/batch/step")
    wandb.define_metric("training/epoch/*", step_metric = "training/epoch/step")
    wandb.define_metric("validation/batch/*", step_metric = "validation/batch/step")
    wandb.define_metric("validation/epoch/*", step_metric = "validation/epoch/step")

    return
