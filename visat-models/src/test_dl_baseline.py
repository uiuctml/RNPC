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

def test(model_baseline, data_loader, device, batch_step):
    utility.loadCheckpointBest(header.config_baseline["dir_checkpoints"], header.config_baseline["file_name_checkpoint_best"], model_baseline)

    accuracy_epoch = 0
    config_dataset = data_loader.dataset.config
    ground_truths_epoch = []
    output_list = []
    predictions_epoch = []
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 0, leave = False)

    model_baseline.eval()
    progress_bar.set_description_str("[INFO]: Testing progress")

    with torch.set_grad_enabled(False):
        for (batch_index, (input, _, labels, _)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)

            output = model_baseline(input)
            (_, predictions) = torch.max(output, 1)

            corrects = torch.sum(predictions == labels.data).item()

            accuracy_batch = corrects / input.size(0)
            accuracy_epoch += corrects

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            wandb.log({"testing/batch/accuracy": accuracy_batch})
            wandb.log({"testing/batch/step": batch_step})

            ground_truths_epoch += labels.data.tolist()
            predictions_epoch += predictions.tolist()

            batch_step += 1

    progress_bar.close()

    accuracy_epoch /= len(data_loader.dataset)
    precision_epoch = sklearn.metrics.precision_score(ground_truths_epoch, predictions_epoch, average = "macro", zero_division = 0)
    recall_epoch = sklearn.metrics.recall_score(ground_truths_epoch, predictions_epoch, average = "macro", zero_division = 0)
    output_list += [accuracy_epoch, precision_epoch, recall_epoch]

    wandb.log({"testing/epoch/accuracy": accuracy_epoch})
    wandb.log({"testing/epoch/precision": precision_epoch})
    wandb.log({"testing/epoch/recall": recall_epoch})
    wandb.summary["testing/epoch/accuracy"] = accuracy_epoch
    wandb.summary["testing/epoch/precision"] = precision_epoch
    wandb.summary["testing/epoch/recall"] = recall_epoch

    logger.log_info("Testing accuracy: " + str(accuracy_epoch) + ".")
    logger.log_trace("Testing precision: " + str(precision_epoch) + ".")
    logger.log_trace("Testing recall: " + str(recall_epoch) + ".")

    utility.logTestOutput(output_list, header.config_baseline, config_dataset)

    return batch_step

def main():
    argument.processArgumentsTestBaseline()

    utility.setSeed(header.config_baseline["seed"])
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    wandb.init(config = header.config_baseline, mode = "disabled")

    dataset_transforms = utility.createTransform(header.config_baseline)
    dataset_test = dataset.VISATDataset(header.config_baseline["dir_dataset_test"], dataset_transforms)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_baseline["data_loader_batch_size"], shuffle = False, num_workers = header.config_baseline["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    model_baseline = model.createModelBaseline(device)
    model_baseline = torch.nn.DataParallel(model_baseline)
    model_baseline = model_baseline.to(device)

    test(model_baseline, data_loader_test, device, 1)

    return

if __name__ == "__main__":
    main()
