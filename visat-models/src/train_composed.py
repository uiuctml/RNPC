#!/usr/bin/env python3

import argument
import composition
import dataset
import header
import logger
import model
import spn
import test_composed
import torch
import torchsummary
import tqdm
import type
import utility
import wandb

def train(model_decomposed, spn_joint, spn_marginal, data_loader, criterion, optimizer_decomposed, optimizer_spn, device, batch_step, marginal_probabilities_counted = None):
    accuracy_epoch_composed = 0
    accuracy_epoch_list_decomposed = []
    loss_epoch = 0
    config_dataset = data_loader.dataset.config
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 1, leave = False)
    spn_output_rows = len(data_loader.dataset.classes_original)
    spn_output_cols = 1

    for attribute in config_dataset["attributes"]:
        attribute_labels = attribute["labels"]
        spn_output_cols *= len(attribute_labels)
        accuracy_epoch_list_decomposed.append(0)

    model_decomposed.train()
    progress_bar.set_description_str("[INFO]: Training progress")

    with torch.set_grad_enabled(True):
        for (batch_index, (input, labels_decomposed, labels_original, _)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels_decomposed = labels_decomposed.to(device, non_blocking = True)
            labels_original = labels_original.to(device, non_blocking = True)

            optimizer_decomposed.zero_grad()

            (outputs_decomposed, _) = model_decomposed(input)

            outputs_decomposed = utility.applySoftmaxDecomposed(outputs_decomposed)
            (matrix_a, matrix_b, output_composed) = composition.Composition.spn(outputs_decomposed, spn_joint, spn_marginal, spn_output_rows, spn_output_cols, device, marginal_probabilities_counted)

            (_, predictions_composed) = torch.max(output_composed, 1)
            loss = criterion(output_composed, labels_original)

            if header.config_decomposed["use_l2_loss"]:
                l2_norm = utility.computeL2Norm(model_decomposed.parameters())
                loss_l2 = header.config_decomposed["l2_lambda"] * l2_norm
                loss += loss_l2

            loss.backward()
            optimizer_decomposed.step()

            spn_joint.backward()

            if marginal_probabilities_counted is None:
                spn_marginal.backward()

            optimizer_spn.step(matrix_a.detach(), matrix_b.detach(), output_composed.detach(), labels_original)

            corrects_composed = torch.sum(predictions_composed == labels_original.data).item()

            for (i, attribute) in enumerate(config_dataset["attributes"]):
                (_, predictions_decomposed) = torch.max(outputs_decomposed[i], 1)

                corrects_decomposed = torch.sum(predictions_decomposed == labels_decomposed[:, i].data).item()
                accuracy_batch_decomposed = corrects_decomposed / input.size(0)
                accuracy_epoch_list_decomposed[i] += corrects_decomposed

                wandb.log({"training/batch/" + attribute["name"] + "/accuracy": accuracy_batch_decomposed})

            accuracy_batch_composed = corrects_composed / input.size(0)
            loss_batch = loss.item()

            accuracy_epoch_composed += corrects_composed
            loss_epoch += loss_batch

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            wandb.log({"training/batch/accuracy": accuracy_batch_composed})
            wandb.log({"training/batch/step": batch_step})
            wandb.log({"training/batch/loss": loss_batch})

            batch_step += 1

    progress_bar.close()

    for (i, attribute) in enumerate(config_dataset["attributes"]):
        accuracy_epoch_list_decomposed[i] /= len(data_loader.dataset)
        wandb.log({"training/epoch/" + attribute["name"] + "/accuracy": accuracy_epoch_list_decomposed[i]})

    accuracy_epoch_composed /= len(data_loader.dataset)
    loss_epoch /= len(data_loader)

    wandb.log({"training/epoch/accuracy": accuracy_epoch_composed})
    wandb.log({"training/epoch/loss": loss_epoch})

    return batch_step

def validate(model_decomposed, spn_joint, spn_marginal, data_loader, criterion, device, batch_step, marginal_probabilities_counted = None):
    accuracy_epoch_composed = 0
    accuracy_epoch_list_decomposed = []
    loss_epoch = 0
    config_dataset = data_loader.dataset.config
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 1, leave = False)
    spn_output_rows = len(data_loader.dataset.classes_original)
    spn_output_cols = 1

    for attribute in config_dataset["attributes"]:
        attribute_labels = attribute["labels"]
        spn_output_cols *= len(attribute_labels)
        accuracy_epoch_list_decomposed.append(0)

    model_decomposed.eval()
    progress_bar.set_description_str("[INFO]: Validation progress")

    with torch.set_grad_enabled(False):
        for (batch_index, (input, labels_decomposed, labels_original, _)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels_decomposed = labels_decomposed.to(device, non_blocking = True)
            labels_original = labels_original.to(device, non_blocking = True)

            (outputs_decomposed, _) = model_decomposed(input)

            outputs_decomposed = utility.applySoftmaxDecomposed(outputs_decomposed)
            (_, _, output_composed) = composition.Composition.spn(outputs_decomposed, spn_joint, spn_marginal, spn_output_rows, spn_output_cols, device, marginal_probabilities_counted)

            (_, predictions_composed) = torch.max(output_composed, 1)
            loss = criterion(output_composed, labels_original)

            if header.config_decomposed["use_l2_loss"]:
                l2_norm = utility.computeL2Norm(model_decomposed.parameters())
                loss_l2 = header.config_decomposed["l2_lambda"] * l2_norm
                loss += loss_l2

            corrects_composed = torch.sum(predictions_composed == labels_original.data).item()

            for (i, attribute) in enumerate(config_dataset["attributes"]):
                (_, predictions_decomposed) = torch.max(outputs_decomposed[i], 1)

                corrects_decomposed = torch.sum(predictions_decomposed == labels_decomposed[:, i].data).item()
                accuracy_batch_decomposed = corrects_decomposed / input.size(0)
                accuracy_epoch_list_decomposed[i] += corrects_decomposed

                wandb.log({"validation/batch/" + attribute["name"] + "/accuracy": accuracy_batch_decomposed})

            accuracy_batch_composed = corrects_composed / input.size(0)
            loss_batch = loss.item()

            accuracy_epoch_composed += corrects_composed
            loss_epoch += loss_batch

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            wandb.log({"validation/batch/accuracy": accuracy_batch_composed})
            wandb.log({"validation/batch/step": batch_step})
            wandb.log({"validation/batch/loss": loss_batch})

            batch_step += 1

    progress_bar.close()

    for (i, attribute) in enumerate(config_dataset["attributes"]):
        accuracy_epoch_list_decomposed[i] /= len(data_loader.dataset)
        wandb.log({"validation/epoch/" + attribute["name"] + "/accuracy": accuracy_epoch_list_decomposed[i]})

    accuracy_epoch_composed /= len(data_loader.dataset)
    loss_epoch /= len(data_loader)

    wandb.log({"validation/epoch/accuracy": accuracy_epoch_composed})
    wandb.log({"validation/epoch/loss": loss_epoch})

    return (accuracy_epoch_composed, loss_epoch, batch_step)

def main():
    resume = argument.processArgumentsTrainComposed()
    model_pretrained_weights = header.config_decomposed["model_pretrained_weights"]

    utility.setSeed(header.seed)
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    if header.run_mode == "online":
        wandb.login()

    config = {
        "decomposed": header.config_decomposed,
        "spn": header.config_spn
    }

    wandb.init(project = header.project_name, name = header.run_name_decomposed, config = config, resume = resume, mode = header.run_mode)

    utility.wAndBDefineMetrics()

    logger.log_info("Started run \"" + header.run_name_decomposed + "\" and " + header.run_name_spn + ".")

    accuracy_validation_best = 0
    batch_step_test = 1
    batch_step_train = 1
    batch_step_validate = 1
    criterion = utility.lossNegativeLogLikelihood
    dataset_transforms = utility.createTransform(header.config_decomposed)
    dataset_test = dataset.VISATDataset(header.config_decomposed["dir_dataset_test"], dataset_transforms)
    dataset_train = dataset.VISATDataset(header.config_decomposed["dir_dataset_train"], dataset_transforms)
    dataset_validation = dataset.VISATDataset(header.config_decomposed["dir_dataset_validation"], dataset_transforms)
    config_dataset = dataset_test.config
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_decomposed["data_loader_batch_size"], shuffle = False, num_workers = header.config_decomposed["data_loader_worker_count"], pin_memory = True)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size = header.config_decomposed["data_loader_batch_size"], shuffle = header.config_decomposed["data_loader_shuffle"], num_workers = header.config_decomposed["data_loader_worker_count"], pin_memory = True)
    data_loader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size = header.config_decomposed["data_loader_batch_size"], shuffle = header.config_decomposed["data_loader_shuffle"], num_workers = header.config_decomposed["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    epoch = 1
    marginal_probabilities_counted = None
    model_decomposed = model.createModelDecomposed(device)
    model_decomposed = torch.nn.DataParallel(model_decomposed)
    model_decomposed = model_decomposed.to(device)
    normalize = False
    progress_bar = None
    spn_joint = spn.SPN(device)
    spn_marginal = spn.SPN(device)
    optimizer_decomposed = torch.optim.SGD(model_decomposed.parameters(), lr = header.config_decomposed["optimizer_learning_rate"], momentum = header.config_decomposed["optimizer_momentum"], weight_decay = header.config_decomposed["optimizer_weight_decay"])
    optimizer_spn = None
    learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_decomposed, header.config_decomposed["learning_rate_scheduler_mode"], header.config_decomposed["learning_rate_scheduler_factor"], header.config_decomposed["learning_rate_scheduler_patience"], header.config_decomposed["learning_rate_scheduler_threshold"], header.config_decomposed["learning_rate_scheduler_threshold_mode"], header.config_decomposed["learning_rate_scheduler_cooldown"], header.config_decomposed["learning_rate_scheduler_min_learning_rate"], header.config_decomposed["learning_rate_scheduler_min_learning_rate_decay"], header.config_decomposed["learning_rate_scheduler_verbose"])
    learning_rate_scheduler_spn = None

    if header.config_spn["optimizer"] == type.OptimizerSPN.cccp_discriminative.name:
        (marginal_probabilities_counted, _) = utility.countAttributeJointProbabilities(config_dataset, device)
        optimizer_spn = spn.CCCPDiscriminativeSPNOptimizer(spn_joint, spn_marginal, device, marginal_probabilities_counted = marginal_probabilities_counted)
    elif header.config_spn["optimizer"] == type.OptimizerSPN.cccp_generative.name:
        optimizer_spn = spn.CCCPGenerativeSPNOptimizer(spn_joint, spn_marginal, device)
    elif header.config_spn["optimizer"] == type.OptimizerSPN.ebw_discriminative.name:
        optimizer_spn = spn.EBWDiscriminativeSPNOptimizer(spn_joint, spn_marginal, device, header.config_spn["optimizer_learning_rate"], header.config_spn["optimizer_prior_factor"], header.config_spn["epsilon_projection"], header.config_spn["growth_threshold"])
    elif header.config_spn["optimizer"] == type.OptimizerSPN.pgd_discriminative.name:
        optimizer_spn = spn.PGDDiscriminativeSPNOptimizer(spn_joint, spn_marginal, device, header.config_spn["optimizer_learning_rate"], header.config_spn["optimizer_prior_factor"], header.config_spn["epsilon_projection"])
        normalize = True
    elif header.config_spn["optimizer"] == type.OptimizerSPN.pgd_generative.name:
        optimizer_spn = spn.PGDGenerativeSPNOptimizer(spn_joint, spn_marginal, device, header.config_spn["optimizer_learning_rate"], header.config_spn["optimizer_prior_factor"], header.config_spn["epsilon_projection"])
        normalize = True
    else:
        logger.log_fatal("Unknown SPN optimizer \"" + header.config_spn["optimizer"] + "\".")
        exit(-1)

    learning_rate_scheduler_spn = spn.LossSPNLearningRateScheduler(optimizer_spn, header.config_spn["learning_rate_scheduler_factor"], header.config_spn["learning_rate_scheduler_patience"], header.config_spn["learning_rate_scheduler_threshold"], header.config_spn["learning_rate_scheduler_cooldown"], header.config_spn["learning_rate_scheduler_min_learning_rate"])

    logger.log_info("Loading SPN from \"" + header.config_spn["file_path_spn"] + "\"...")

    spn_joint.load(header.config_spn["file_path_spn"])
    spn_marginal.load(header.config_spn["file_path_spn"])
    optimizer_spn.set_weights_prior(spn_joint.get_weights())

    logger.log_info("Loading SPN leaf node settings...")

    spn_settings_joint = utility.generateSPNSettings(config_dataset, device)
    spn_settings_marginal = torch.clone(spn_settings_joint)
    spn_settings_marginal[:, -1] = -1

    logger.log_info("Setting SPN leaf nodes...")

    spn_joint.set_leaf_nodes(spn_settings_joint)
    spn_marginal.set_leaf_nodes(spn_settings_marginal)

    (accuracy_validation_best, batch_step_train, batch_step_validate, [criterion], epoch) = utility.loadCheckpoint(header.config_decomposed["dir_checkpoints"], header.config_decomposed["file_name_checkpoint"], accuracy_validation_best, batch_step_train, batch_step_validate, [criterion], epoch, [learning_rate_scheduler], model_decomposed, [optimizer_decomposed])
    utility.loadCheckpointSPN(spn_joint, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint"], 0, 0, 0)
    utility.loadCheckpointSPN(spn_marginal, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint"], 0, 0, 0)

    if header.config_decomposed["fine_tuning"]:
        utility.loadCheckpointBest(header.config_decomposed["dir_checkpoints"], model_pretrained_weights, model_decomposed)

    if header.config_spn["fine_tuning"]:
        utility.loadCheckpointBestSPN(spn_joint, header.config_spn["dir_checkpoints"], header.config_spn["model_pretrained_weights"])
        utility.loadCheckpointBestSPN(spn_marginal, header.config_spn["dir_checkpoints"], header.config_spn["model_pretrained_weights"])
    elif header.config_spn["randomize_weights"]:
        logger.log_info("Randomizing SPN weights...")
        spn_joint.randomize_weights()
        spn_marginal.set_weights(spn_joint.get_weights())

    if header.show_model_summary:
        model_input_size = (header.config_decomposed["model_input_channels"], header.config_decomposed["model_input_height"], header.config_decomposed["model_input_width"])
        torchsummary.summary(model_decomposed, input_size = model_input_size)

        logger.log_info("Number of nodes: " + str(len(spn_joint.nodes)) + ".")
        logger.log_info("Number of sum nodes: " + str(len(spn_joint.sum_nodes)) + ".")
        logger.log_info("Number of product nodes: " + str(len(spn_joint.product_nodes)) + ".")
        logger.log_info("Number of leaf nodes: " + str(len(spn_joint.leaf_nodes)) + ".")
        logger.log_info("SPN depths: " + str(spn_joint.depth) + ".")
        logger.log_info("SPN leaf node setting dimension: (" + str(int(spn_settings_joint.shape[0])) + ", " + str(int(spn_settings_joint.shape[1])) + ").")

    if epoch <= header.config_decomposed["epochs"]:
        progress_bar = tqdm.tqdm(total = header.config_decomposed["epochs"], position = 0)
        progress_bar.set_description_str("[INFO]: Epoch")

    while epoch <= header.config_decomposed["epochs"]:
        if progress_bar is not None:
            progress_bar.n = epoch
            progress_bar.refresh()

        wandb.log({"training/epoch/step": epoch})
        wandb.log({"validation/epoch/step": epoch})

        batch_step_train = train(model_decomposed, spn_joint, spn_marginal, data_loader_train, criterion, optimizer_decomposed, optimizer_spn, device, batch_step_train, marginal_probabilities_counted)
        (accuracy_validation_epoch, loss_validation_epoch, batch_step_validate) = validate(model_decomposed, spn_joint, spn_marginal, data_loader_validation, criterion, device, batch_step_validate, marginal_probabilities_counted)

        learning_rate_scheduler.step(loss_validation_epoch)
        learning_rate_scheduler_spn.step(loss_validation_epoch)

        if accuracy_validation_epoch > accuracy_validation_best:
            accuracy_validation_best = accuracy_validation_epoch
            wandb.log({"validation/epoch/accuracy_best": accuracy_validation_best})
            utility.saveCheckpoint(header.config_decomposed["dir_checkpoints"], header.config_decomposed["file_name_checkpoint_best"], accuracy_validation_best, batch_step_train, batch_step_validate, [criterion], epoch, [learning_rate_scheduler], model_decomposed, [optimizer_decomposed])
            utility.saveCheckpointSPN(spn_joint, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint_best"], 0, 0, 0)

        utility.saveCheckpoint(header.config_decomposed["dir_checkpoints"], header.config_decomposed["file_name_checkpoint"], accuracy_validation_best, batch_step_train, batch_step_validate, [criterion], epoch, [learning_rate_scheduler], model_decomposed, [optimizer_decomposed])
        utility.saveCheckpointSPN(spn_joint, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint"], 0, 0, 0)

        epoch += 1

    if progress_bar is not None:
        progress_bar.close()

    logger.log_info("Best validation accuracy: " + str(accuracy_validation_best) + ".")
    wandb.summary["validation/epoch/accuracy_best"] = accuracy_validation_best

    if normalize:
        settings_marginal = torch.full((1, spn_settings_joint.shape[1]), -1).to(device)
        logger.log_info("Normalizing SPN weights...")

        utility.loadCheckpointBestSPN(spn_marginal, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint"])
        spn_marginal(settings_marginal)
        spn_marginal.normalize_weights(header.config_spn["epsilon_smoothing"])
        utility.saveCheckpointSPN(spn_marginal, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint"], 0, 0, 0)

        utility.loadCheckpointBestSPN(spn_marginal, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint_best"])
        spn_marginal(settings_marginal)
        spn_marginal.normalize_weights(header.config_spn["epsilon_smoothing"])
        utility.saveCheckpointSPN(spn_marginal, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint_best"], 0, 0, 0)

        spn_marginal.set_leaf_nodes(spn_settings_marginal)

    wandb.log({"testing/epoch/step": batch_step_test})
    test_composed.test(model_decomposed, spn_joint, spn_marginal, spn_settings_joint, data_loader_test, device, batch_step_test, marginal_probabilities_counted)

    wandb.finish()

    return

if __name__ == "__main__":
    main()
