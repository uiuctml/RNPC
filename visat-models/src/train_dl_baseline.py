#!/usr/bin/env python3

import argument
import dataset
import header
import logger
import model
import test_dl_baseline
import torch
import torchsummary
import tqdm
import utility
import wandb

def train(model_baseline, data_loader, criterion, optimizer, device, batch_step):
    accuracy_epoch = 0
    loss_epoch = 0
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 1, leave = False)

    model_baseline.train()
    progress_bar.set_description_str("[INFO]: Training progress")

    with torch.set_grad_enabled(True):
        for (batch_index, (input, _, labels, _)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)

            optimizer.zero_grad()

            output = model_baseline(input)
            (_, predictions) = torch.max(output, 1)
            loss = criterion(output, labels)

            if header.config_baseline["use_l2_loss"]:
                l2_norm = utility.computeL2Norm(model_baseline.parameters())
                loss_l2 = header.config_baseline["l2_lambda"] * l2_norm
                loss += loss_l2

            loss.backward()
            optimizer.step()

            corrects = torch.sum(predictions == labels.data).item()

            accuracy_batch = corrects / input.size(0)
            loss_batch = loss.item()

            accuracy_epoch += corrects
            loss_epoch += loss_batch

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            wandb.log({"training/batch/accuracy": accuracy_batch})
            wandb.log({"training/batch/step": batch_step})
            wandb.log({"training/batch/loss": loss_batch})

            batch_step += 1

    progress_bar.close()

    accuracy_epoch /= len(data_loader.dataset)
    loss_epoch /= len(data_loader)

    wandb.log({"training/epoch/accuracy": accuracy_epoch})
    wandb.log({"training/epoch/loss": loss_epoch})

    return batch_step

def validate(model_baseline, data_loader, criterion, device, batch_step):
    accuracy_epoch = 0
    loss_epoch = 0
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 1, leave = False)

    model_baseline.eval()
    progress_bar.set_description_str("[INFO]: Validation progress")

    with torch.set_grad_enabled(False):
        for (batch_index, (input, _, labels, _)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)

            output = model_baseline(input)
            (_, predictions) = torch.max(output, 1)
            loss = criterion(output, labels)

            if header.config_baseline["use_l2_loss"]:
                l2_norm = utility.computeL2Norm(model_baseline.parameters())
                loss_l2 = header.config_baseline["l2_lambda"] * l2_norm
                loss += loss_l2

            corrects = torch.sum(predictions == labels.data).item()

            accuracy_batch = corrects / input.size(0)
            loss_batch = loss.item()

            accuracy_epoch += corrects
            loss_epoch += loss_batch

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            wandb.log({"validation/batch/accuracy": accuracy_batch})
            wandb.log({"validation/batch/step": batch_step})
            wandb.log({"validation/batch/loss": loss_batch})

            batch_step += 1

    progress_bar.close()

    accuracy_epoch /= len(data_loader.dataset)
    loss_epoch /= len(data_loader)

    wandb.log({"validation/epoch/accuracy": accuracy_epoch})
    wandb.log({"validation/epoch/loss": loss_epoch})

    return (accuracy_epoch, loss_epoch, batch_step)

def main():
    resume = argument.processArgumentsTrainBaseline()

    utility.setSeed(header.config_baseline["seed"])
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    if header.run_mode == "online":
        wandb.login()

    wandb.init(project = header.project_name, name = header.run_name_baseline, config = header.config_baseline, resume = resume, mode = header.run_mode)

    utility.wAndBDefineMetrics()

    logger.log_info("Started run \"" + header.run_name_baseline + "\".")

    accuracy_validation_best = 0
    batch_step_test = 1
    batch_step_train = 1
    batch_step_validate = 1
    criterion = torch.nn.CrossEntropyLoss()
    dataset_transforms = utility.createTransform(header.config_baseline)
    dataset_test = dataset.VISATDataset(header.config_baseline["dir_dataset_test"], dataset_transforms)
    dataset_train = dataset.VISATDataset(header.config_baseline["dir_dataset_train"], dataset_transforms)
    dataset_validation = dataset.VISATDataset(header.config_baseline["dir_dataset_validation"], dataset_transforms)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_baseline["data_loader_batch_size"], shuffle = False, num_workers = header.config_baseline["data_loader_worker_count"], pin_memory = True)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size = header.config_baseline["data_loader_batch_size"], shuffle = header.config_baseline["data_loader_shuffle"], num_workers = header.config_baseline["data_loader_worker_count"], pin_memory = True)
    data_loader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size = header.config_baseline["data_loader_batch_size"], shuffle = header.config_baseline["data_loader_shuffle"], num_workers = header.config_baseline["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    epoch = 1
    model_baseline = model.createModelBaseline(device)
    model_baseline = torch.nn.DataParallel(model_baseline)
    model_baseline = model_baseline.to(device)
    optimizer = torch.optim.SGD(model_baseline.module.get_parameters(), lr = header.config_baseline["optimizer_learning_rate"], momentum = header.config_baseline["optimizer_momentum"], weight_decay = header.config_baseline["optimizer_weight_decay"])
    learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, header.config_baseline["learning_rate_scheduler_mode"], header.config_baseline["learning_rate_scheduler_factor"], header.config_baseline["learning_rate_scheduler_patience"], header.config_baseline["learning_rate_scheduler_threshold"], header.config_baseline["learning_rate_scheduler_threshold_mode"], header.config_baseline["learning_rate_scheduler_cooldown"], header.config_baseline["learning_rate_scheduler_min_learning_rate"], header.config_baseline["learning_rate_scheduler_min_learning_rate_decay"], header.config_baseline["learning_rate_scheduler_verbose"])
    progress_bar = None

    (accuracy_validation_best, batch_step_train, batch_step_validate, [criterion], epoch) = utility.loadCheckpoint(header.config_baseline["dir_checkpoints"], header.config_baseline["file_name_checkpoint"], accuracy_validation_best, batch_step_train, batch_step_validate, [criterion], epoch, [learning_rate_scheduler], model_baseline, [optimizer])

    if header.show_model_summary:
        model_input_size = (header.config_baseline["model_input_channels"], header.config_baseline["model_input_height"], header.config_baseline["model_input_width"])
        torchsummary.summary(model_baseline, input_size = model_input_size)

    if epoch <= header.config_baseline["epochs"]:
        progress_bar = tqdm.tqdm(total = header.config_baseline["epochs"], position = 0)
        progress_bar.set_description_str("[INFO]: Epoch")

    while epoch <= header.config_baseline["epochs"]:
        if progress_bar is not None:
            progress_bar.n = epoch
            progress_bar.refresh()

        wandb.log({"training/epoch/step": epoch})
        wandb.log({"validation/epoch/step": epoch})

        batch_step_train = train(model_baseline, data_loader_train, criterion, optimizer, device, batch_step_train)
        (accuracy_validation_epoch, loss_validation_epoch, batch_step_validate) = validate(model_baseline, data_loader_validation, criterion, device, batch_step_validate)

        learning_rate_scheduler.step(loss_validation_epoch)

        if accuracy_validation_epoch > accuracy_validation_best:
            accuracy_validation_best = accuracy_validation_epoch
            wandb.log({"validation/epoch/accuracy_best": accuracy_validation_best})
            utility.saveCheckpoint(header.config_baseline["dir_checkpoints"], header.config_baseline["file_name_checkpoint_best"], accuracy_validation_best, batch_step_train, batch_step_validate, [criterion], epoch, [learning_rate_scheduler], model_baseline, [optimizer])

        utility.saveCheckpoint(header.config_baseline["dir_checkpoints"], header.config_baseline["file_name_checkpoint"], accuracy_validation_best, batch_step_train, batch_step_validate, [criterion], epoch, [learning_rate_scheduler], model_baseline, [optimizer])

        epoch += 1

    if progress_bar is not None:
        progress_bar.close()

    logger.log_info("Best validation accuracy: " + str(accuracy_validation_best) + ".")
    wandb.summary["validation/epoch/accuracy_best"] = accuracy_validation_best

    wandb.log({"testing/epoch/step": batch_step_test})
    test_dl_baseline.test(model_baseline, data_loader_test, device, batch_step_test)

    wandb.finish()

    return

if __name__ == "__main__":
    main()
