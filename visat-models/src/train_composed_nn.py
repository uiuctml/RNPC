#!/usr/bin/env python3

import argument
import dataset
import header
import logger
import model
import test_composed_nn
import torch
import tqdm
import utility
import wandb

def train(model_decomposed, model_relation_nn, data_loader, criterion, optimizer_decomposed, device, batch_step):
    accuracy_epoch_composed = 0
    accuracy_epoch_list_decomposed = []
    loss_epoch = 0
    config_dataset = data_loader.dataset.config
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 1, leave = False)

    for _ in config_dataset["attributes"]:
        accuracy_epoch_list_decomposed.append(0)

    model_decomposed.train()
    model_relation_nn.train()
    progress_bar.set_description_str("[INFO]: Training progress")

    with torch.set_grad_enabled(True):
        for (batch_index, (input, labels_decomposed, labels_original, _)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels_decomposed = labels_decomposed.to(device, non_blocking = True)
            labels_original = labels_original.to(device, non_blocking = True)

            optimizer_decomposed.zero_grad()

            (outputs_decomposed, _) = model_decomposed(input)
            output_composed = model_relation_nn(outputs_decomposed)

            (_, predictions_composed) = torch.max(output_composed, 1)
            loss = criterion(output_composed, labels_original)

            if header.config_decomposed["use_l2_loss"]:
                l2_norm = utility.computeL2Norm(model_decomposed.parameters())
                loss_l2 = header.config_decomposed["l2_lambda"] * l2_norm
                loss += loss_l2

            loss.backward()
            optimizer_decomposed.step()

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

def validate(model_decomposed, model_relation_nn, data_loader, criterion, device, batch_step):
    accuracy_epoch_composed = 0
    accuracy_epoch_list_decomposed = []
    loss_epoch = 0
    config_dataset = data_loader.dataset.config
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 1, leave = False)

    for _ in config_dataset["attributes"]:
        accuracy_epoch_list_decomposed.append(0)

    model_decomposed.eval()
    model_relation_nn.eval()
    progress_bar.set_description_str("[INFO]: Validation progress")

    with torch.set_grad_enabled(False):
        for (batch_index, (input, labels_decomposed, labels_original, _)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels_decomposed = labels_decomposed.to(device, non_blocking = True)
            labels_original = labels_original.to(device, non_blocking = True)

            (outputs_decomposed, _) = model_decomposed(input)
            output_composed = model_relation_nn(outputs_decomposed)

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
    header.run_name_relation = header.run_name_decomposed.replace(header.run_name_decomposed_keyword, header.run_name_relation_keyword)
    file_name_checkpoint_relation = header.run_name_relation + ".tar"
    file_name_checkpoint_best_relation = header.run_name_relation + ".best.tar"
    model_pretrained_weights = header.config_decomposed["model_pretrained_weights"]

    utility.setSeed(header.seed)
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    if header.run_mode == "online":
        wandb.login()

    wandb.init(project = header.project_name, name = header.run_name_decomposed, config = header.config_decomposed, resume = resume, mode = header.run_mode)

    utility.wAndBDefineMetrics()

    logger.log_info("Started run \"" + header.run_name_decomposed + "\" and " + header.run_name_relation + ".")

    accuracy_validation_best = 0
    batch_step_test = 1
    batch_step_train = 1
    batch_step_validate = 1
    criterion =torch.nn.CrossEntropyLoss()
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
    model_decomposed = model.createModelDecomposed(device)
    model_decomposed = torch.nn.DataParallel(model_decomposed)
    model_decomposed = model_decomposed.to(device)
    model_relation_nn = model.RelationNN(config_dataset, device)
    model_relation_nn = torch.nn.DataParallel(model_relation_nn)
    model_relation_nn = model_relation_nn.to(device)
    progress_bar = None
    optimizer_decomposed = torch.optim.SGD(list(model_decomposed.parameters()) + list(model_relation_nn.parameters()), lr = header.config_decomposed["optimizer_learning_rate"], momentum = header.config_decomposed["optimizer_momentum"], weight_decay = header.config_decomposed["optimizer_weight_decay"])
    learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_decomposed, header.config_decomposed["learning_rate_scheduler_mode"], header.config_decomposed["learning_rate_scheduler_factor"], header.config_decomposed["learning_rate_scheduler_patience"], header.config_decomposed["learning_rate_scheduler_threshold"], header.config_decomposed["learning_rate_scheduler_threshold_mode"], header.config_decomposed["learning_rate_scheduler_cooldown"], header.config_decomposed["learning_rate_scheduler_min_learning_rate"], header.config_decomposed["learning_rate_scheduler_min_learning_rate_decay"], header.config_decomposed["learning_rate_scheduler_verbose"])

    (accuracy_validation_best, batch_step_train, batch_step_validate, [criterion], epoch) = utility.loadCheckpoint(header.config_decomposed["dir_checkpoints"], header.config_decomposed["file_name_checkpoint"], accuracy_validation_best, batch_step_train, batch_step_validate, [criterion], epoch, [learning_rate_scheduler], model_decomposed, [optimizer_decomposed])
    utility.loadCheckpoint(header.config_decomposed["dir_checkpoints"], file_name_checkpoint_relation, accuracy_validation_best, batch_step_train, batch_step_validate, [criterion], epoch, [], model_relation_nn, [])

    utility.loadCheckpointBest(header.config_decomposed["dir_checkpoints"], model_pretrained_weights, model_decomposed)

    if epoch <= header.config_decomposed["epochs"]:
        progress_bar = tqdm.tqdm(total = header.config_decomposed["epochs"], position = 0)
        progress_bar.set_description_str("[INFO]: Epoch")

    while epoch <= header.config_decomposed["epochs"]:
        if progress_bar is not None:
            progress_bar.n = epoch
            progress_bar.refresh()

        wandb.log({"training/epoch/step": epoch})
        wandb.log({"validation/epoch/step": epoch})

        batch_step_train = train(model_decomposed, model_relation_nn, data_loader_train, criterion, optimizer_decomposed, device, batch_step_train)
        (accuracy_validation_epoch, loss_validation_epoch, batch_step_validate) = validate(model_decomposed, model_relation_nn, data_loader_validation, criterion, device, batch_step_validate)

        learning_rate_scheduler.step(loss_validation_epoch)

        if accuracy_validation_epoch > accuracy_validation_best:
            accuracy_validation_best = accuracy_validation_epoch
            wandb.log({"validation/epoch/accuracy_best": accuracy_validation_best})
            utility.saveCheckpoint(header.config_decomposed["dir_checkpoints"], header.config_decomposed["file_name_checkpoint_best"], accuracy_validation_best, batch_step_train, batch_step_validate, [criterion], epoch, [learning_rate_scheduler], model_decomposed, [optimizer_decomposed])
            utility.saveCheckpoint(header.config_decomposed["dir_checkpoints"], file_name_checkpoint_best_relation, 0, 0, 0, [], 0, [], model_relation_nn, [])

        utility.saveCheckpoint(header.config_decomposed["dir_checkpoints"], header.config_decomposed["file_name_checkpoint"], accuracy_validation_best, batch_step_train, batch_step_validate, [criterion], epoch, [learning_rate_scheduler], model_decomposed, [optimizer_decomposed])
        utility.saveCheckpoint(header.config_decomposed["dir_checkpoints"], file_name_checkpoint_relation, 0, 0, 0, [], 0, [], model_relation_nn, [])

        epoch += 1

    if progress_bar is not None:
        progress_bar.close()

    logger.log_info("Best validation accuracy: " + str(accuracy_validation_best) + ".")
    wandb.summary["validation/epoch/accuracy_best"] = accuracy_validation_best

    wandb.log({"testing/epoch/step": batch_step_test})
    test_composed_nn.test(model_decomposed, model_relation_nn, data_loader_test, device, batch_step_test)

    wandb.finish()

    return

if __name__ == "__main__":
    main()
