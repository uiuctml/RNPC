import pdb

import header
import json
import logger
import os
import PIL.Image
import torch
import utility

class VISATDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform = None):
        self.class_to_idx = []
        self.class_to_idx_original = {}
        self.classes = []
        self.classes_original = []
        self.config = {}
        self.file_paths = []
        self.labels = []
        self.labels_original = []
        self.root = root
        self.transform = transform

        if not os.path.isdir(root):
            logger.log_error("Invalid dataset directory.")
            exit(-1)

        logger.log_info("Loading dataset \"" + root + "\"...")

        file_config_dataset = open(header.file_path_dataset_config, "r")
        self.config = json.load(file_config_dataset)
        file_config_dataset.close()

        self.classes = utility.getLabelsAttribute(self.config)
        self.classes_original = utility.getLabelsOriginal(self.config)
        self.class_to_idx = utility.getIndicesFromLabelsAttribute(self.classes)
        self.class_to_idx_original = utility.getIndicesFromLabelsOriginal(self.classes_original)

        for _ in self.config["attributes"]:
            self.labels.append([])

        if "instance_wise" in self.config and self.config["instance_wise"]:
            for class_name_original in os.listdir(root):
                for file_name in os.listdir(os.path.join(root, class_name_original)):
                    image_name = os.path.join(class_name_original, file_name)
                    label_original = self.class_to_idx_original[class_name_original]

                    self.file_paths.append(os.path.abspath(os.path.join(root, image_name)))
                    self.labels_original.append(label_original)

                    for (attribute_index, attribute) in enumerate(self.config["attributes"]):
                        attribute_name = attribute["name"]
                        class_name_decomposed = self.config["mappings"][image_name]["labels"][attribute_name]
                        label = self.class_to_idx[attribute_name][class_name_decomposed]
                        self.labels[attribute_index].append(label)
        else:
            for class_name_original in self.classes_original:
                for file_name in os.listdir(os.path.join(root, class_name_original)):
                    label_original = self.class_to_idx_original[class_name_original]

                    self.file_paths.append(os.path.abspath(os.path.join(root, class_name_original, file_name)))
                    self.labels_original.append(label_original)

                    for (attribute_index, attribute) in enumerate(self.config["attributes"]):
                        attribute_name = attribute["name"]
                        class_name_decomposed = self.config["mappings"][class_name_original]["labels"][attribute_name]
                        label = self.class_to_idx[attribute_name][class_name_decomposed]
                        self.labels[attribute_index].append(label)

        return

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        image = PIL.Image.open(self.file_paths[index]).convert("RGB")
        labels = []

        if self.transform is not None:
            image = self.transform(image)

        for label in self.labels:
            labels.append(int(label[index]))

        return (image, torch.LongTensor(labels), self.labels_original[index], self.file_paths[index])
