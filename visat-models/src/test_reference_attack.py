#!/usr/bin/env python3
import argument
import csv
import dataset
import header
import logger
import model
import os
import sklearn.metrics
import torch
import tqdm
import type
import utility
import wandb
import pdb

# NEW: import necessary packages
import sys
sys.path.append('../adversarial-attacks-pytorch')
from torchattacks import PGD, PGDL2, CW, CWBS

def computeAccuracy(output_neck, output_head, labels_decomposed, labels_original, data_loader, device):
    labels_attribute = data_loader.dataset.classes

    accuracy_attribute_batch = None
    accuracy_task_batch = None
    threshold_accuracy_attribute = 0.5
    threshold_accuracy_task = 0.5

    if header.config_reference["model"] == type.ModelReference.cbm.name:
        threshold_accuracy_attribute = 0
        threshold_accuracy_task = 0
    elif header.config_reference["model"] == type.ModelReference.cbm_cat.name or header.config_reference["model"] == type.ModelReference.cem.name:
        threshold_accuracy_task = 0

    # if header.config_reference["model"] == type.ModelReference.cbm_cat.name:
    #     accuracy_attribute_batch = utility.computeAccuracyDecomposed(output_neck, labels_decomposed, device)
    # else:
    # labels_decomposed = utility.getBinaryLabelsDecomposed(labels_decomposed)

    outputs_decomposed = (output_neck > threshold_accuracy_attribute)
    # accuracy_attribute_batch = sklearn.metrics.accuracy_score(labels_decomposed.cpu(), outputs_decomposed.cpu())
    # Compute accuracies for attributes
    attribute_corrects = []
    start_idx = 0
    for col_idx, (attribute, values) in enumerate(labels_attribute.items()):
        num_values = len(values)

        # Extract relevant columns for the current attribute
        labels = labels_decomposed[:, start_idx:start_idx+num_values]
        outputs = outputs_decomposed[:, start_idx:start_idx+num_values]

        # Check if all elements in a row match
        correct_attribute_pred = (labels == outputs).all(dim=1).float()  # Row-wise correctness
        attribute_corrects.append(correct_attribute_pred.sum().item())

        start_idx += num_values  # Move start index for the next attribute
    assert len(attribute_corrects)==len(labels_attribute), "len(accuracies) should be equal to len(labels_attribute)."

    outputs_original = (output_head > threshold_accuracy_task)
    # accuracy_task_batch = sklearn.metrics.accuracy_score(labels_original.cpu(), outputs_original.cpu())
    correct_class_pred = (labels_original == outputs_original).all(dim=1).float()
    class_corrects = correct_class_pred.sum().item()

    return (attribute_corrects, class_corrects)

def test(model_reference, data_loader, device, batch_step):
    utility.loadCheckpointBest(header.config_reference["dir_checkpoints"], header.config_reference["file_name_checkpoint_best"], model_reference)
    csv_results = {}

    accuracy_attribute_epoch_list = []
    accuracy_task_epoch = 0
    config_dataset = data_loader.dataset.config
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 0, leave = False)

    for _ in config_dataset["attributes"]:
        accuracy_attribute_epoch_list.append(0)

    model_reference.eval()
    progress_bar.set_description_str("[INFO]: Testing progress")

    # NEW: configure attack ids and type
    attack_ids = torch.tensor(header.config_reference["attack_ids"]).to(device, non_blocking=True)

    if header.config_reference["attack_name"] == "pgd":
        atk = PGD(model_reference, eps=header.config_reference["attack_bound"], alpha=2/255, steps=50, random_start=True)
    elif header.config_reference["attack_name"] == "pgdl2":
        atk = PGDL2(model_reference, eps=header.config_reference["attack_bound"], alpha=header.config_reference["attack_bound"]/10, steps=50, random_start=True)
    elif header.config_reference["attack_name"] == "cw":
        atk = CWBS(model_reference, init_c=1.0, kappa=0, steps=10, lr=0.01, binary_search_steps=int(header.config_reference["attack_bound"]))
    else:
        raise ValueError(f"Unknown attack name: {header.config_reference['attack_name']}")

    print(atk)
    adv_norm_list = []

    # NEW: set grad to be True
    with torch.set_grad_enabled(True):
        for (batch_index, (input, labels_decomposed, labels_original, _)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels_original = labels_original.to(device, non_blocking = True)
            labels_decomposed = labels_decomposed.to(device, non_blocking=True)

            # NEW: add perturbations on inputs
            if header.config_reference["attack_name"] == "cw":
                adv_images = atk(input, labels_decomposed, attack_ids, data_loader)
                original_flat = input.view(input.size(0), -1)
                perturbed_flat = adv_images.view(adv_images.size(0), -1)
                adv_norm = (torch.sqrt(torch.sum((original_flat - perturbed_flat) ** 2, dim=1))).mean().item()
            else:
                adv_images = atk(input, labels_decomposed, attack_ids, data_loader)
                adv_norm = header.config_reference["attack_bound"]
            adv_norm_list.append(adv_norm)

            labels_original_binary = utility.getBinaryLabelsOriginal(labels_original, data_loader)
            labels_decomposed_binary = utility.getBinaryLabelsDecomposed(labels_decomposed, data_loader)
            labels_decomposed_binary = labels_decomposed_binary.to(device, non_blocking=True)

            (output_neck, output_head) = model_reference(adv_images)

            (accuracy_attribute_batch, accuracy_task_batch) = computeAccuracy(output_neck, output_head, labels_decomposed_binary, labels_original_binary, data_loader, device)

            accuracy_attribute_epoch_list = [accuracy_attribute_epoch_list[i] + accuracy_attribute_batch[i] for i in range(len(accuracy_attribute_epoch_list))]
            accuracy_task_epoch += accuracy_task_batch

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            for (i, dataset_entry) in enumerate(config_dataset["attributes"]):
                wandb.log({"testing/batch/" + dataset_entry["name"] + "/accuracy": accuracy_attribute_batch[i] / input.size(0)})
            wandb.log({"testing/batch/accuracy_task": accuracy_task_batch / input.size(0)})
            wandb.log({"testing/batch/step": batch_step})

            batch_step += 1

    progress_bar.close()

    csv_results["attack_name"] = header.config_reference["attack_name"]
    csv_results["attack_ids"] = " ".join(str(id) for id in header.config_reference["attack_ids"])
    csv_results["attack_norm"] = round(sum(adv_norm_list) / len(adv_norm_list), 2)

    for (i, dataset_entry) in enumerate(config_dataset["attributes"]):
        wandb.log({"testing/epoch/" + dataset_entry["name"] + "/accuracy": accuracy_attribute_epoch_list[i] / len(data_loader.dataset)})
        wandb.summary["testing/epoch/accuracy_attribute"] = accuracy_attribute_epoch_list[i] / len(data_loader.dataset)
        logger.log_info(f"Testing {dataset_entry['name']} accuracy: " + str(accuracy_attribute_epoch_list[i] / len(data_loader.dataset)) + ".")
        csv_results[dataset_entry["name"]] = accuracy_attribute_epoch_list[i] / len(data_loader.dataset)

    accuracy_task_epoch /= len(data_loader.dataset)
    wandb.log({"testing/epoch/accuracy_task": accuracy_task_epoch})
    wandb.summary["testing/epoch/accuracy_task"] = accuracy_task_epoch
    logger.log_info("Testing task accuracy: " + str(accuracy_task_epoch) + ".")
    csv_results["class"] = accuracy_task_epoch

    # Write csv results
    fieldnames = csv_results.keys()
    if "cbm" in header.config_reference['run_name']:
        csv_dir = os.path.join(header.config_decomposed["dir_results"], "test_reference_attack_cbm")
    elif "dcr" in header.config_reference['run_name']:
        csv_dir = os.path.join(header.config_decomposed["dir_results"], "test_reference_attack_dcr")
    os.makedirs(csv_dir, exist_ok=True)
    csv_file = os.path.join(csv_dir, f"{csv_results['attack_name']}_{csv_results['attack_ids']}_{header.config_reference['attack_bound']}.csv")
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(csv_results)

    return batch_step

def main():
    argument.processArgumentsTestReference()

    utility.setSeed(header.config_reference["seed"])
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    wandb.init(config = header.config_reference, mode = "disabled")

    dataset_transforms = utility.createTransform(header.config_reference)
    dataset_test = dataset.VISATDataset(header.config_reference["dir_dataset_test"], dataset_transforms)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_reference["data_loader_batch_size"], shuffle = False, num_workers = header.config_reference["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    model_reference = model.createModelReference(device)
    model_reference = torch.nn.DataParallel(model_reference)
    model_reference = model_reference.to(device)

    test(model_reference, data_loader_test, device, 1)

    return

if __name__ == "__main__":
    main()
