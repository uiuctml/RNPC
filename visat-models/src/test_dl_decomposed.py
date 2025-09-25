#!/usr/bin/env python3

import argument
import dataset
import header
import logger
import model
import sklearn.metrics
import torch
import tqdm
import utility
import wandb

def test(model_decomposed, data_loader, device, batch_step):
    utility.loadCheckpointBest(header.config_decomposed["dir_checkpoints"], header.config_decomposed["file_name_checkpoint_best"], model_decomposed)

    accuracy_epoch_list = []
    config_dataset = data_loader.dataset.config
    ground_truths_epoch_list = []
    output_list = []
    output_list_accuracy = []
    output_list_precision = []
    output_list_recall = []
    predictions_epoch_list = []
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 0, leave = False)

    for _ in config_dataset["attributes"]:
        accuracy_epoch_list.append(0)
        ground_truths_epoch_list.append([])
        predictions_epoch_list.append([])

    model_decomposed.eval()
    progress_bar.set_description_str("[INFO]: Testing progress")

    with torch.set_grad_enabled(False):
        for (batch_index, (input, labels, _, _)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)

            (outputs, _) = model_decomposed(input)

            for (i, dataset_entry) in enumerate(config_dataset["attributes"]):
                (_, predictions) = torch.max(outputs[i], 1)

                corrects = torch.sum(predictions == labels[:, i].data).item()
                accuracy_batch = corrects / input.size(0)
                accuracy_epoch_list[i] += corrects

                wandb.log({"testing/batch/" + dataset_entry["name"] + "/accuracy": accuracy_batch})

                ground_truths_epoch_list[i] += labels[:, i].data.tolist()
                predictions_epoch_list[i] += predictions.tolist()

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            wandb.log({"testing/batch/step": batch_step})

            batch_step += 1

    progress_bar.close()

    for (i, dataset_entry) in enumerate(config_dataset["attributes"]):
        accuracy_epoch_list[i] /= len(data_loader.dataset)
        precision_epoch = sklearn.metrics.precision_score(ground_truths_epoch_list[i], predictions_epoch_list[i], average = "macro", zero_division = 0)
        recall_epoch = sklearn.metrics.recall_score(ground_truths_epoch_list[i], predictions_epoch_list[i], average = "macro", zero_division = 0)

        output_list_accuracy.append(accuracy_epoch_list[i])
        output_list_precision.append(precision_epoch)
        output_list_recall.append(recall_epoch)

        wandb.log({"testing/epoch/" + dataset_entry["name"] + "/accuracy": accuracy_epoch_list[i]})
        wandb.log({"testing/epoch/" + dataset_entry["name"] + "/precision": precision_epoch})
        wandb.log({"testing/epoch/" + dataset_entry["name"] + "/recall": recall_epoch})
        wandb.summary["testing/epoch/" + dataset_entry["name"] + "/accuracy"] = accuracy_epoch_list[i]
        wandb.summary["testing/epoch/" + dataset_entry["name"] + "/precision"] = precision_epoch
        wandb.summary["testing/epoch/" + dataset_entry["name"] + "/recall"] = recall_epoch

        logger.log_info("Testing accuracy for \"" + dataset_entry["name"] + "\": " + str(accuracy_epoch_list[i]) + ".")
        logger.log_trace("Testing precision for \"" + dataset_entry["name"] + "\": " + str(precision_epoch) + ".")
        logger.log_trace("Testing recall for \"" + dataset_entry["name"] + "\": " + str(recall_epoch) + ".")

    output_list += output_list_accuracy
    output_list += output_list_precision
    output_list += output_list_recall

    utility.logTestOutput(output_list, header.config_decomposed, config_dataset)

    return batch_step

def main():
    argument.processArgumentsTestDecomposed()

    utility.setSeed(header.config_decomposed["seed"])
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    wandb.init(config = header.config_decomposed, mode = "disabled")

    dataset_transforms = utility.createTransform(header.config_decomposed)
    dataset_test = dataset.VISATDataset(header.config_decomposed["dir_dataset_test"], dataset_transforms)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_decomposed["data_loader_batch_size"], shuffle = False, num_workers = header.config_decomposed["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    model_decomposed = model.createModelDecomposed(device)
    model_decomposed = torch.nn.DataParallel(model_decomposed)
    model_decomposed = model_decomposed.to(device)

    test(model_decomposed, data_loader_test, device, 1)

    return

if __name__ == "__main__":
    main()
