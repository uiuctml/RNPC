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

def test(model_decomposed, model_relation_nn, data_loader, device, batch_step):
    file_name_checkpoint_best_relation = header.run_name_relation + ".best.tar"

    utility.loadCheckpointBest(header.config_decomposed["dir_checkpoints"], header.config_decomposed["file_name_checkpoint_best"], model_decomposed)
    utility.loadCheckpointBest(header.config_decomposed["dir_checkpoints"], file_name_checkpoint_best_relation, model_relation_nn)

    accuracy_epoch_composed = 0
    accuracy_epoch_list_decomposed = []
    config_dataset = data_loader.dataset.config
    ground_truths_epoch_composed = []
    ground_truths_epoch_list_decomposed = []
    output_list_composed = []
    output_list_decomposed = []
    output_list_decomposed_accuracy = []
    output_list_decomposed_precision = []
    output_list_decomposed_recall = []
    predictions_epoch_composed = []
    predictions_epoch_list_decomposed = []
    progress_bar = tqdm.tqdm(total = len(data_loader), position = 0, leave = False)

    for _ in config_dataset["attributes"]:
        accuracy_epoch_list_decomposed.append(0)
        ground_truths_epoch_list_decomposed.append([])
        predictions_epoch_list_decomposed.append([])

    model_decomposed.eval()
    model_relation_nn.eval()
    progress_bar.set_description_str("[INFO]: Testing progress")

    with torch.set_grad_enabled(False):
        for (batch_index, (input, labels_decomposed, labels_original, _)) in enumerate(data_loader):
            input = input.to(device, non_blocking = True)
            labels_decomposed = labels_decomposed.to(device, non_blocking = True)
            labels_original = labels_original.to(device, non_blocking = True)

            (outputs_decomposed, _) = model_decomposed(input)

            for (i, dataset_entry) in enumerate(config_dataset["attributes"]):
                (_, predictions_decomposed) = torch.max(outputs_decomposed[i], 1)

                corrects_decomposed = torch.sum(predictions_decomposed == labels_decomposed[:, i].data).item()
                accuracy_batch_decomposed = corrects_decomposed / input.size(0)
                accuracy_epoch_list_decomposed[i] += corrects_decomposed

                wandb.log({"testing/batch/" + dataset_entry["name"] + "/accuracy": accuracy_batch_decomposed})

                ground_truths_epoch_list_decomposed[i] += labels_decomposed[:, i].data.tolist()
                predictions_epoch_list_decomposed[i] += predictions_decomposed.tolist()

            output_composed = model_relation_nn(outputs_decomposed)

            (_, predictions_composed) = torch.max(output_composed, 1)

            corrects_composed = torch.sum(predictions_composed == labels_original.data).item()

            accuracy_batch_composed = corrects_composed / input.size(0)
            accuracy_epoch_composed += corrects_composed

            progress_bar.n = batch_index + 1
            progress_bar.refresh()

            wandb.log({"testing/batch/accuracy": accuracy_batch_composed})
            wandb.log({"testing/batch/step": batch_step})

            ground_truths_epoch_composed += labels_original.data.tolist()
            predictions_epoch_composed += predictions_composed.tolist()

            batch_step += 1

    progress_bar.close()

    for (i, dataset_entry) in enumerate(config_dataset["attributes"]):
        accuracy_epoch_list_decomposed[i] /= len(data_loader.dataset)
        precision_epoch_decomposed = sklearn.metrics.precision_score(ground_truths_epoch_list_decomposed[i], predictions_epoch_list_decomposed[i], average = "macro", zero_division = 0)
        recall_epoch_decomposed = sklearn.metrics.recall_score(ground_truths_epoch_list_decomposed[i], predictions_epoch_list_decomposed[i], average = "macro", zero_division = 0)

        output_list_decomposed_accuracy.append(accuracy_epoch_list_decomposed[i])
        output_list_decomposed_precision.append(precision_epoch_decomposed)
        output_list_decomposed_recall.append(recall_epoch_decomposed)

        wandb.log({"testing/epoch/" + dataset_entry["name"] + "/accuracy": accuracy_epoch_list_decomposed[i]})
        wandb.log({"testing/epoch/" + dataset_entry["name"] + "/precision": precision_epoch_decomposed})
        wandb.log({"testing/epoch/" + dataset_entry["name"] + "/recall": recall_epoch_decomposed})
        wandb.summary["testing/epoch/" + dataset_entry["name"] + "/accuracy"] = accuracy_epoch_list_decomposed[i]
        wandb.summary["testing/epoch/" + dataset_entry["name"] + "/precision"] = precision_epoch_decomposed
        wandb.summary["testing/epoch/" + dataset_entry["name"] + "/recall"] = recall_epoch_decomposed

        logger.log_info("Decomposed testing accuracy for \"" + dataset_entry["name"] + "\": " + str(accuracy_epoch_list_decomposed[i]) + ".")
        logger.log_trace("Decomposed testing precision for \"" + dataset_entry["name"] + "\": " + str(precision_epoch_decomposed) + ".")
        logger.log_trace("Decomposed testing recall for \"" + dataset_entry["name"] + "\": " + str(recall_epoch_decomposed) + ".")

    accuracy_epoch_composed /= len(data_loader.dataset)

    precision_epoch_composed = sklearn.metrics.precision_score(ground_truths_epoch_composed, predictions_epoch_composed, average = "macro", zero_division = 0)
    recall_epoch_composed = sklearn.metrics.recall_score(ground_truths_epoch_composed, predictions_epoch_composed, average = "macro", zero_division = 0)
    output_list_composed += [accuracy_epoch_composed, precision_epoch_composed, recall_epoch_composed]
    output_list_decomposed += output_list_decomposed_accuracy
    output_list_decomposed += output_list_decomposed_precision
    output_list_decomposed += output_list_decomposed_recall

    wandb.log({"testing/epoch/accuracy": accuracy_epoch_composed})
    wandb.log({"testing/epoch/precision": precision_epoch_composed})
    wandb.log({"testing/epoch/recall": recall_epoch_composed})
    wandb.summary["testing/epoch/accuracy"] = accuracy_epoch_composed
    wandb.summary["testing/epoch/precision"] = precision_epoch_composed
    wandb.summary["testing/epoch/recall"] = recall_epoch_composed

    logger.log_info("Composed testing accuracy: " + str(accuracy_epoch_composed) + ".")
    logger.log_trace("Composed testing precision: " + str(precision_epoch_composed) + ".")
    logger.log_trace("Composed testing recall: " + str(recall_epoch_composed) + ".")

    utility.logTestOutput(output_list_composed, header.config_baseline, config_dataset, True)
    utility.logTestOutput(output_list_decomposed, header.config_decomposed, config_dataset)

    return

def main():
    argument.processArgumentsTestDecomposed()

    header.run_name_relation = header.run_name_decomposed.replace(header.run_name_decomposed_keyword, header.run_name_relation_keyword)

    utility.setSeed(header.seed)
    torch.backends.cuda.matmul.allow_tf32 = header.cuda_allow_tf32

    wandb.init(config = header.config_decomposed, mode = "disabled")

    dataset_transforms = utility.createTransform(header.config_decomposed)
    dataset_test = dataset.VISATDataset(header.config_decomposed["dir_dataset_test"], dataset_transforms)
    config_dataset = dataset_test.config
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = header.config_decomposed["data_loader_batch_size"], shuffle = False, num_workers = header.config_decomposed["data_loader_worker_count"], pin_memory = True)
    device = torch.device("cuda")
    model_decomposed = model.createModelDecomposed(device)
    model_decomposed = torch.nn.DataParallel(model_decomposed)
    model_decomposed = model_decomposed.to(device)
    model_relation_nn = model.RelationNN(config_dataset, device)
    model_relation_nn = torch.nn.DataParallel(model_relation_nn)
    model_relation_nn = model_relation_nn.to(device)

    test(model_decomposed, model_relation_nn, data_loader_test, device, 1)

    return

if __name__ == "__main__":
    main()
