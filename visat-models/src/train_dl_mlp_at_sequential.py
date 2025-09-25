#!/usr/bin/env python3

import argument
import dataset
import header
import logger
import model
import test_dl_decomposed
import torch
import torchsummary
import tqdm
import utility
import wandb

# NEW: import necessary packages
import sys
sys.path.append('../adversarial-attacks-pytorch')
from torchattacks import PGD, PGDL2, CW, CWBS, AutoAttack, Square

import pdb

def train(model_decomposed, data_loader, criterions, optimizers, device, batch_step, atk):
    accuracy_epoch_list = []
    config_dataset = data_loader.dataset.config
    loss_epoch_list = []
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 1, leave = False)

    for _ in config_dataset["attributes"]:
        accuracy_epoch_list.append(0)
        loss_epoch_list.append(0)

    model_decomposed.train()
    progress_bar.set_description_str("[INFO]: Training progress")

    # NEW: configure attack ids and type
    attack_ids_all = torch.tensor(header.config_decomposed["attack_ids"]).to(device, non_blocking=True)
    adv_norm_list = []

    with torch.set_grad_enabled(True):
        for (batch_index, (input, labels, _, _)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)

            # New: support randomly select an attack id per batch
            attack_ids = attack_ids_all[torch.randint(0, len(attack_ids_all), (1,))].to(device)

            # NEW: add perturbations on inputs
            if header.config_decomposed["attack_name"] == "cw":
                adv_images = atk(input, labels, attack_ids)
                original_flat = input.view(input.size(0), -1)
                perturbed_flat = adv_images.view(adv_images.size(0), -1)
                adv_norm = (torch.sqrt(torch.sum((original_flat - perturbed_flat) ** 2, dim=1))).mean().item()
            else:
                adv_images = atk(input, labels, attack_ids)
                adv_norm = header.config_decomposed["attack_bound"]
            adv_norm_list.append(adv_norm)

            outputs_clean, _ = model_decomposed(input)
            outputs_adv, _ = model_decomposed(adv_images)

            parameters = model_decomposed.module.get_parameters()

            for (i, dataset_entry) in enumerate(config_dataset["attributes"]):
                optimizer = optimizers[i]
                optimizer.zero_grad()

                # Clean and adversarial loss
                loss_clean = criterions[i](outputs_clean[i], labels[:, i])
                loss_adv = criterions[i](outputs_adv[i], labels[:, i])
                loss = 0.5 * (loss_clean + loss_adv)

                # Optional L2 regularization
                if header.config_decomposed["use_l2_loss"]:
                    l2_norm = utility.computeL2Norm(parameters[i])
                    loss_l2 = header.config_decomposed["l2_lambda"] * l2_norm
                    loss += loss_l2

                loss.backward()
                optimizer.step()

                # Accuracy computed on adversarial examples
                _, predictions = torch.max(outputs_adv[i], 1)
                corrects = torch.sum(predictions == labels[:, i].data).item()
                accuracy_batch = corrects / input.size(0)
                loss_batch = loss.item()

                accuracy_epoch_list[i] += corrects
                loss_epoch_list[i] += loss_batch

                wandb.log({f"training/batch/{dataset_entry['name']}/accuracy": accuracy_batch})
                wandb.log({f"training/batch/{dataset_entry['name']}/loss": loss_batch})

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            wandb.log({"training/batch/step": batch_step})

            batch_step += 1

    progress_bar.close()

    for (i, dataset_entry) in enumerate(config_dataset["attributes"]):
        accuracy_epoch_list[i] /= len(data_loader.dataset)
        loss_epoch_list[i] /= len(data_loader)
        wandb.log({"training/epoch/" + dataset_entry["name"] + "/accuracy": accuracy_epoch_list[i]})
        wandb.log({"training/epoch/" + dataset_entry["name"] + "/loss": loss_epoch_list[i]})

    return batch_step

def validate(model_decomposed, data_loader, criterions, device, batch_step):
    accuracy_epoch_list = []
    config_dataset = data_loader.dataset.config
    loss_epoch_list = []
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 1, leave = False)

    for _ in config_dataset["attributes"]:
        accuracy_epoch_list.append(0)
        loss_epoch_list.append(0)

    model_decomposed.eval()
    progress_bar.set_description_str("[INFO]: Validation progress")

    with torch.set_grad_enabled(False):
        for (batch_index, (input, labels, _, _)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)

            (outputs, _) = model_decomposed(input)
            parameters = model_decomposed.module.get_parameters()

            for (i, dataset_entry) in enumerate(config_dataset["attributes"]):
                (_, predictions) = torch.max(outputs[i], 1)
                loss = criterions[i](outputs[i], labels[:, i])

                if header.config_decomposed["use_l2_loss"]:
                    l2_norm = utility.computeL2Norm(parameters[i])
                    loss_l2 = header.config_decomposed["l2_lambda"] * l2_norm
                    loss += loss_l2

                corrects = torch.sum(predictions == labels[:, i].data).item()

                accuracy_batch = corrects / input.size(0)
                loss_batch = loss.item()

                accuracy_epoch_list[i] += corrects
                loss_epoch_list[i] += loss_batch

                wandb.log({"validation/batch/" + dataset_entry["name"] + "/accuracy": accuracy_batch})
                wandb.log({"validation/batch/" + dataset_entry["name"] + "/loss": loss_batch})

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            wandb.log({"validation/batch/step": batch_step})

            batch_step += 1

    progress_bar.close()

    for (i, dataset_entry) in enumerate(config_dataset["attributes"]):
        accuracy_epoch_list[i] /= len(data_loader.dataset)
        loss_epoch_list[i] /= len(data_loader)
        wandb.log({"validation/epoch/" + dataset_entry["name"] + "/accuracy": accuracy_epoch_list[i]})
        wandb.log({"validation/epoch/" + dataset_entry["name"] + "/loss": loss_epoch_list[i]})

    return (accuracy_epoch_list, loss_epoch_list, batch_step)

def main():
    resume = argument.processArgumentsTrainDecomposed()

    utility.setSeed(header.config_decomposed["seed"])
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    if header.run_mode == "online":
        wandb.login()

    wandb.init(project = header.project_name, name = header.run_name_decomposed, config = header.config_decomposed, resume = resume, mode = header.run_mode)

    utility.wAndBDefineMetrics()

    logger.log_info("Started run \"" + header.run_name_decomposed + "\".")

    accuracy_validation_best = 0
    batch_step_test = 1
    batch_step_train = 1
    batch_step_validate = 1
    criterions = []
    dataset_transforms = utility.createTransform(header.config_decomposed)
    dataset_test = dataset.VISATDataset(header.config_decomposed["dir_dataset_test"], dataset_transforms)
    dataset_train = dataset.VISATDataset(header.config_decomposed["dir_dataset_train"], dataset_transforms)
    dataset_validation = dataset.VISATDataset(header.config_decomposed["dir_dataset_validation"], dataset_transforms)
    config_dataset = dataset_train.config
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_decomposed["data_loader_batch_size"], shuffle = False, num_workers = header.config_decomposed["data_loader_worker_count"], pin_memory = True)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size = header.config_decomposed["data_loader_batch_size"], shuffle = header.config_decomposed["data_loader_shuffle"], num_workers = header.config_decomposed["data_loader_worker_count"], pin_memory = True)
    data_loader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size = header.config_decomposed["data_loader_batch_size"], shuffle = header.config_decomposed["data_loader_shuffle"], num_workers = header.config_decomposed["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    epoch = 1
    learning_rate_schedulers = []
    model_decomposed = model.createModelDecomposed(device)
    model_decomposed = torch.nn.DataParallel(model_decomposed)
    model_decomposed = model_decomposed.to(device)

    # NEW: configure attack ids and type
    if header.config_decomposed["attack_name"] == "pgd":
        atk = PGD(model_decomposed, eps=header.config_decomposed["attack_bound"], alpha=2 / 255, steps=50,
                  random_start=True)
    elif header.config_decomposed["attack_name"] == "pgdl2":
        atk = PGDL2(model_decomposed, eps=header.config_decomposed["attack_bound"],
                    alpha=header.config_decomposed["attack_bound"] / 10, steps=50, random_start=True)
    elif header.config_decomposed["attack_name"] == "cw":
        atk = CWBS(model_decomposed, init_c=1.0, kappa=0, steps=10, lr=0.01,
                   binary_search_steps=int(header.config_decomposed["attack_bound"]))
    elif header.config_decomposed["attack_name"] == "square":
        atk = Square(model_decomposed, norm='Linf', eps=header.config_decomposed["attack_bound"], n_queries=1000,
                     n_restarts=1, p_init=0.8, seed=0, verbose=False, loss='margin', resc_schedule=True)
    else:
        raise ValueError(f"Unknown attack name: {header.config_decomposed['attack_name']}")

    if header.config_decomposed["targeted_attack"]:
        atk.set_mode_targeted_by_label()  # New for targeted attack
    print(atk)

    optimizers = []
    parameters = model_decomposed.module.get_parameters()
    progress_bar = None

    for i in range(len(config_dataset["attributes"])):
        optimizer = torch.optim.SGD(parameters[i], lr = header.config_decomposed["optimizer_learning_rate"], momentum = header.config_decomposed["optimizer_momentum"], weight_decay = header.config_decomposed["optimizer_weight_decay"])

        learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, header.config_decomposed["learning_rate_scheduler_mode"], header.config_decomposed["learning_rate_scheduler_factor"], header.config_decomposed["learning_rate_scheduler_patience"], header.config_decomposed["learning_rate_scheduler_threshold"], header.config_decomposed["learning_rate_scheduler_threshold_mode"], header.config_decomposed["learning_rate_scheduler_cooldown"], header.config_decomposed["learning_rate_scheduler_min_learning_rate"], header.config_decomposed["learning_rate_scheduler_min_learning_rate_decay"], header.config_decomposed["learning_rate_scheduler_verbose"])

        learning_rate_schedulers.append(learning_rate_scheduler)
        optimizers.append(optimizer)
        criterions.append(torch.nn.CrossEntropyLoss())

    (accuracy_validation_best, batch_step_train, batch_step_validate, criterions, epoch) = utility.loadCheckpoint(header.config_decomposed["dir_checkpoints"], header.config_decomposed["file_name_checkpoint"], accuracy_validation_best, batch_step_train, batch_step_validate, criterions, epoch, learning_rate_schedulers, model_decomposed, optimizers)

    if header.show_model_summary:
        model_input_size = (header.config_decomposed["model_input_channels"], header.config_decomposed["model_input_height"], header.config_decomposed["model_input_width"])
        torchsummary.summary(model_decomposed, input_size = model_input_size)

    if epoch <= header.config_decomposed["epochs"]:
        progress_bar = tqdm.tqdm(total = header.config_decomposed["epochs"], position = 0)
        progress_bar.set_description_str("[INFO]: Epoch")

    while epoch <= header.config_decomposed["epochs"]:
        if progress_bar is not None:
            progress_bar.n = epoch
            progress_bar.refresh()

        wandb.log({"training/epoch/step": epoch})
        wandb.log({"validation/epoch/step": epoch})

        batch_step_train = train(model_decomposed, data_loader_train, criterions, optimizers, device, batch_step_train, atk)
        (accuracy_validation_epoch_list, loss_validation_epoch_list, batch_step_validate) = validate(model_decomposed, data_loader_validation, criterions, device, batch_step_validate)

        for (learning_rate_scheduler, loss_validation_epoch) in zip(learning_rate_schedulers, loss_validation_epoch_list):
            learning_rate_scheduler.step(loss_validation_epoch)

        accuracy_validation_epoch_mean = sum(accuracy_validation_epoch_list) / len(accuracy_validation_epoch_list)

        if accuracy_validation_epoch_mean > accuracy_validation_best:
            accuracy_validation_best = accuracy_validation_epoch_mean
            wandb.log({"validation/epoch/accuracy_best": accuracy_validation_best})
            utility.saveCheckpoint(header.config_decomposed["dir_checkpoints"], header.config_decomposed["file_name_checkpoint_best"], accuracy_validation_best, batch_step_train, batch_step_validate, criterions, epoch, learning_rate_schedulers, model_decomposed, optimizers)

        utility.saveCheckpoint(header.config_decomposed["dir_checkpoints"], header.config_decomposed["file_name_checkpoint"], accuracy_validation_best, batch_step_train, batch_step_validate, criterions, epoch, learning_rate_schedulers, model_decomposed, optimizers)

        epoch += 1

    if progress_bar is not None:
        progress_bar.close()

    logger.log_info("Best validation accuracy: " + str(accuracy_validation_best) + ".")
    wandb.summary["validation/epoch/accuracy_best"] = accuracy_validation_best

    wandb.log({"testing/epoch/step": batch_step_test})
    test_dl_decomposed.test(model_decomposed, data_loader_test, device, batch_step_test)

    wandb.finish()

    return

if __name__ == "__main__":
    main()
