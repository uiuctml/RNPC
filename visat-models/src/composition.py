import header
import logger
import torch

class Composition():
    def __init__(self, dataset, device):
        self.class_indices_decomposed = []
        self.dataset = dataset
        self.device = device

        self.gatherDecomposedClassIndices()

        return

    def gatherDecomposedClassIndices(self):
        for class_name_original in self.dataset.classes_original:
            class_indices_decomposed = []

            for dataset in self.dataset.config["attributes"]:
                dataset_name = dataset["name"]
                class_name_decomposed = self.dataset.config["mappings"][class_name_original]["labels"][dataset_name]

                if class_name_decomposed == "":
                    logger.log_warn("\"" + class_name_original + "\" contains an empty label for dataset \"" + dataset_name + "\".")
                    class_name_decomposed = dataset_name + header.config_decomposed["dataset_delimiter_label"] + header.config_decomposed["dataset_label_undefined_keyword"]

                class_indices_decomposed.append(self.dataset.class_to_idx[dataset_name][class_name_decomposed])

            self.class_indices_decomposed.append(class_indices_decomposed)

        self.class_indices_decomposed = torch.LongTensor(self.class_indices_decomposed)
        self.class_indices_decomposed = self.class_indices_decomposed.to(self.device)

        return

    def gatherDecomposedPredictionConfidences(self, outputs_decomposed):
        batch_size = outputs_decomposed[0].size(0)
        outputs_decomposed_gathered = torch.Tensor()
        outputs_decomposed_gathered = outputs_decomposed_gathered.to(self.device)

        for task_index in range(0, len(outputs_decomposed)):
            class_indices_task = self.class_indices_decomposed[:, task_index]
            class_indices_task = class_indices_task.repeat(1, batch_size)
            class_indices_task = class_indices_task.view(batch_size, len(self.dataset.classes_original), -1)

            outputs_decomposed_task = outputs_decomposed[task_index].repeat(1, len(self.dataset.classes_original))
            outputs_decomposed_task = outputs_decomposed_task.view(batch_size, len(self.dataset.classes_original), -1)

            outputs_task = torch.gather(outputs_decomposed_task, 2, class_indices_task)
            outputs_decomposed_gathered = torch.cat([outputs_decomposed_gathered, outputs_task], 2)

        return outputs_decomposed_gathered

    def naive_bayes(self, outputs_decomposed):
        outputs_decomposed_gathered = self.gatherDecomposedPredictionConfidences(outputs_decomposed)
        output_composed = torch.prod(outputs_decomposed_gathered, 2)
        return output_composed

    @staticmethod
    def spn(outputs_decomposed, spn_joint, spn_marginal, spn_output_rows, spn_output_cols, device, marginal_probabilities_counted = None):
        log_likelihoods_joint = spn_joint.forward()

        if marginal_probabilities_counted is None:
            log_likelihoods_marginal = spn_marginal.forward()

            # Compute matrix A and set entries with zero joint and marginal probabilities to zero
            mask_joint = (log_likelihoods_joint == -float("inf"))
            mask_marginal = (log_likelihoods_joint == -float("inf"))
            mask_matrix_a = mask_joint & mask_marginal
            matrix_a = torch.exp(log_likelihoods_joint - log_likelihoods_marginal)
            matrix_a[mask_matrix_a] = 0
        else:
            matrix_a = torch.exp(log_likelihoods_joint - torch.log(marginal_probabilities_counted))

        matrix_a = matrix_a.reshape(spn_output_rows, spn_output_cols)

        batch_size = outputs_decomposed[0].shape[0]
        matrix_b_list = []

        for batch in range(batch_size):
            matrix_b_batch = outputs_decomposed[0][batch]

            for task_index in range(1, len(outputs_decomposed)):
                matrix_b_batch = torch.outer(matrix_b_batch, outputs_decomposed[task_index][batch]).flatten()

            matrix_b_list.append(matrix_b_batch)

        matrix_b = torch.stack(matrix_b_list, dim = 0).t()
        matrix_b = matrix_b.to(device)

        matrix_c = torch.matmul(matrix_a, matrix_b).t()

        return (matrix_a, matrix_b, matrix_c)

    @staticmethod
    def spn_filter(outputs_decomposed, spn_joint, spn_marginal, spn_output_rows, spn_output_cols, device, marginal_probabilities_counted=None):
        log_likelihoods_joint = spn_joint.forward()

        if marginal_probabilities_counted is None:
            log_likelihoods_marginal = spn_marginal.forward()

            # Compute matrix A and set entries with zero joint and marginal probabilities to zero
            mask_joint = (log_likelihoods_joint == -float("inf"))
            mask_marginal = (log_likelihoods_joint == -float("inf"))
            mask_matrix_a = mask_joint & mask_marginal
            matrix_a = torch.exp(log_likelihoods_joint - log_likelihoods_marginal)
            matrix_a[mask_matrix_a] = 0
        else:
            matrix_a = torch.exp(log_likelihoods_joint - torch.log(marginal_probabilities_counted))

        matrix_a = matrix_a.reshape(spn_output_rows, spn_output_cols)

        batch_size = outputs_decomposed[0].shape[0]
        matrix_b_list = []

        for batch in range(batch_size):
            matrix_b_batch = outputs_decomposed[0][batch]

            for task_index in range(1, len(outputs_decomposed)):
                matrix_b_batch = torch.outer(matrix_b_batch, outputs_decomposed[task_index][batch]).flatten()

            matrix_b_list.append(matrix_b_batch)

        matrix_b = torch.stack(matrix_b_list, dim=0).t()
        matrix_b = matrix_b.to(device)

        matrix_c = torch.matmul(matrix_a, matrix_b).t()

        return (matrix_a, matrix_b, matrix_c)

    @staticmethod
    def spn_single(batch, outputs_decomposed, spn_joint, spn_marginal, spn_output_rows, spn_output_cols, device):
        log_likelihoods_joint = spn_joint.forward()
        log_likelihoods_marginal = spn_marginal.forward()

        # Compute matrix A and set entries with zero joint and marginal probabilities to zero
        mask_joint = (log_likelihoods_joint == -float("inf"))
        mask_marginal = (log_likelihoods_joint == -float("inf"))
        mask_matrix_a = mask_joint & mask_marginal
        matrix_a = torch.exp(log_likelihoods_joint - log_likelihoods_marginal)
        matrix_a[mask_matrix_a] = 0
        matrix_a = matrix_a.reshape(spn_output_rows, spn_output_cols)

        matrix_b_batch = outputs_decomposed[0][batch]

        for task_index in range(1, len(outputs_decomposed)):
            matrix_b_batch = torch.outer(matrix_b_batch, outputs_decomposed[task_index][batch]).flatten()

        matrix_b_batch = matrix_b_batch.unsqueeze(0).to(device).t()
        matrix_c = torch.matmul(matrix_a, matrix_b_batch).t().squeeze(0)

        return (matrix_a, matrix_b_batch, matrix_c)
