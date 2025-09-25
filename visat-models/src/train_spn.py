#!/usr/bin/env python3

import argument
import header
import logger
import spn
import torch
import tqdm
import type
import utility
import wandb

def loadDataset(file_path_dataset, device):
    dataset = []

    with open(file_path_dataset, "r") as file_dataset:
        lines = file_dataset.readlines()

        for line in lines:
            line = line.strip()
            line_list = line.split(",")

            for i in range(len(line_list)):
                line_list[i] = int(line_list[i])

            dataset.append(line_list)

    dataset = torch.Tensor(dataset)
    dataset = dataset.to(device)

    return dataset

def test(spn_joint, settings):
    log_likelihoods = spn_joint(settings)
    log_likelihood_test = torch.mean(log_likelihoods).item()

    logger.log_info("Testing log likelihood: " + str(log_likelihood_test) + ".")

    return

def train(spn_joint, spn_marginal, settings_joint, settings_marginal, optimizer, use_probability):
    log_likelihoods = spn_joint(settings_joint)
    log_likelihoods_marginal = spn_marginal(settings_marginal)

    spn_joint.backward()
    spn_marginal.backward()

    optimizer.step()

    if use_probability:
        log_likelihoods -= log_likelihoods_marginal

    return torch.mean(log_likelihoods).item()

def validate(spn_joint, spn_marginal, settings_joint, settings_marginal, use_probability):
    log_likelihoods = spn_joint(settings_joint)
    log_likelihoods_marginal = spn_marginal(settings_marginal)

    if use_probability:
        log_likelihoods -= log_likelihoods_marginal

    return torch.mean(log_likelihoods).item()

def main():
    resume = argument.processArgumentsTrainSPN()

    utility.setSeed(header.config_spn["seed"])
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    wandb.init(project = header.project_name, name = header.run_name_spn, config = header.config_spn, resume = resume, mode = header.run_mode)

    utility.wAndBDefineMetrics()

    logger.log_info("Started run \"" + header.run_name_spn + "\".")

    device = torch.device("cuda")
    dataset_test = loadDataset(header.config_spn["file_path_spn_dataset_test"], device)
    dataset_train = loadDataset(header.config_spn["file_path_spn_dataset_train"], device)
    dataset_validation = loadDataset(header.config_spn["file_path_spn_dataset_validation"], device)
    epoch = 1
    log_likelihood_best = float("-inf")
    log_likelihood_train = 0
    log_likelihood_train_last = 0
    normalize = False
    use_probability = False
    settings_marginal = torch.full((1, dataset_test.shape[1]), -1).to(device)
    spn_joint = spn.SPN(device)
    spn_marginal = spn.SPN(device)
    optimizer = None
    progress_bar = None

    if header.config_spn["optimizer"] == type.OptimizerSPN.cccp_generative.name:
        optimizer = spn.CCCPGenerativeSPNOptimizer(spn_joint, spn_marginal, device)
    elif header.config_spn["optimizer"] == type.OptimizerSPN.pgd_generative.name:
        optimizer = spn.PGDGenerativeSPNOptimizer(spn_joint, spn_marginal, device, header.config_spn["optimizer_learning_rate"], header.config_spn["optimizer_prior_factor"], header.config_spn["epsilon_projection"])
        normalize = True
        use_probability = True
    else:
        logger.log_fatal("Unknown or unsupported SPN optimizer \"" + header.config_spn["optimizer"] + "\".")
        exit(-1)

    learning_rate_scheduler = spn.LikelihoodSPNLearningRateScheduler(optimizer, header.config_spn["learning_rate_scheduler_factor"])

    logger.log_info("Loading SPN from \"" + header.config_spn["file_path_spn"] + "\"...")

    spn_joint.load(header.config_spn["file_path_spn"])
    spn_marginal.load(header.config_spn["file_path_spn"])
    optimizer.set_weights_prior(spn_joint.get_weights())

    (log_likelihood_best, log_likelihood_train_last, epoch) = utility.loadCheckpointSPN(spn_joint, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint"], log_likelihood_best, log_likelihood_train_last, epoch)
    log_likelihood_train = log_likelihood_train_last

    if header.config_spn["fine_tuning"]:
        utility.loadCheckpointBestSPN(spn_joint, header.config_spn["dir_checkpoints"], header.config_spn["model_pretrained_weights"])
        utility.loadCheckpointBestSPN(spn_marginal, header.config_spn["dir_checkpoints"], header.config_spn["model_pretrained_weights"])
    elif header.config_spn["randomize_weights"]:
        logger.log_info("Randomizing SPN weights...")
        spn_joint.randomize_weights()
        spn_marginal.set_weights(spn_joint.get_weights())

    if header.show_model_summary:
        logger.log_info("Number of nodes: " + str(len(spn_joint.nodes)) + ".")
        logger.log_info("Number of sum nodes: " + str(len(spn_joint.sum_nodes)) + ".")
        logger.log_info("Number of product nodes: " + str(len(spn_joint.product_nodes)) + ".")
        logger.log_info("Number of leaf nodes: " + str(len(spn_joint.leaf_nodes)) + ".")
        logger.log_info("SPN depths: " + str(spn_joint.depth) + ".")

    test(spn_joint, dataset_test)

    if epoch <= header.config_spn["epochs"]:
        progress_bar = tqdm.tqdm(total = header.config_spn["epochs"], position = 0)
        progress_bar.set_description_str("[INFO]: Epoch")

    while epoch <= header.config_spn["epochs"]:
        if progress_bar is not None:
            progress_bar.n = epoch
            progress_bar.refresh()

        log_likelihood_train_last = log_likelihood_train
        log_likelihood_train = train(spn_joint, spn_marginal, dataset_train, settings_marginal, optimizer, use_probability)
        log_likelihood_validate = validate(spn_joint, spn_marginal, dataset_validation, settings_marginal, use_probability)

        if use_probability:
            logger.log_info("Training log probability: " + str(log_likelihood_train) + ".")
            logger.log_info("Validation log probability: " + str(log_likelihood_validate) + ".")
        else:
            logger.log_info("Training log likelihood: " + str(log_likelihood_train) + ".")
            logger.log_info("Validation log likelihood: " + str(log_likelihood_validate) + ".")

        learning_rate_scheduler.step(log_likelihood_train)

        if log_likelihood_validate > log_likelihood_best:
            log_likelihood_best = log_likelihood_validate
            utility.saveCheckpointSPN(spn_joint, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint_best"], log_likelihood_best, log_likelihood_train_last, epoch)

        utility.saveCheckpointSPN(spn_joint, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint"], log_likelihood_best, log_likelihood_train_last, epoch)

        if epoch > 1 and abs(log_likelihood_train - log_likelihood_train_last) < header.config_spn["stopping_criterion"]:
            logger.log_info("Stopping criterion reached.")
            break

        epoch += 1

    if progress_bar is not None:
        progress_bar.close()

    if normalize:
        wandb.run.resumed = True

        logger.log_info("Normalizing SPN weights...")

        utility.loadCheckpointSPN(spn_marginal, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint"], 0, 0, 0)
        spn_marginal(settings_marginal)
        spn_marginal.normalize_weights(header.config_spn["epsilon_smoothing"])
        utility.saveCheckpointSPN(spn_marginal, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint"], log_likelihood_best, log_likelihood_train_last, epoch)

        (log_likelihood_best, log_likelihood_train_last, epoch) = utility.loadCheckpointSPN(spn_marginal, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint_best"], log_likelihood_best, log_likelihood_train_last, epoch)
        spn_marginal(settings_marginal)
        spn_marginal.normalize_weights(header.config_spn["epsilon_smoothing"])
        utility.saveCheckpointSPN(spn_marginal, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint_best"], log_likelihood_best, log_likelihood_train_last, epoch)

        wandb.run.resumed = False

    utility.loadCheckpointBestSPN(spn_joint, header.config_spn["dir_checkpoints"], header.config_spn["file_name_checkpoint_best"])
    test(spn_joint, dataset_test)

    return

if __name__=="__main__":
    main()
