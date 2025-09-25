import abc
import pdb

import header
import json
import logger
import torch
import torchvision
import type
import utility
import torch_explain

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        return

    @abc.abstractmethod
    def forward(self, input):
        pass

    @abc.abstractmethod
    def get_parameters(self):
        pass

class CBM(Model):
    def __init__(self, config_dataset, device):
        super().__init__()

        labels_attribute = utility.getLabelsAttribute(config_dataset)
        labels_original = utility.getLabelsOriginal(config_dataset)
        labels_categories = []

        for attribute_name in labels_attribute.keys():
            labels_categories += labels_attribute[attribute_name]

        input_size = header.config_decomposed["model_input_height"] * header.config_decomposed["model_input_width"] * header.config_decomposed["model_input_channels"]
        hidden_size = header.config_decomposed["head_hidden_size"] * len(config_dataset["attributes"])
        output_size = len(labels_categories)

        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size)
        )

        self.head = torch.nn.Linear(output_size, len(labels_original))

    def forward(self, input):
        output_neck = self.net(input)  # Pass through the Sequential model
        output_head = self.head(output_neck)  # Pass through the head layer

        return (output_neck, output_head)

    def get_parameters(self):
        return list(self.net.parameters()) + list(self.head.parameters())

class CNNMTL(Model):
    def __init__(self, config_dataset, device):
        super().__init__()

        conv_channel_size = 10
        conv_filter_size = 3
        max_pool_size = 2
        input_size = header.config_decomposed["model_input_height"]
        conv_output_size = (input_size - (conv_filter_size - 1) - (conv_filter_size - 1)) // max_pool_size
        linear_input_size = conv_output_size * conv_output_size * conv_channel_size

        self.neck = torch.nn.Sequential(
            torch.nn.Conv2d(header.config_decomposed["model_input_channels"], conv_channel_size, conv_filter_size),
            torch.nn.Conv2d(conv_channel_size, conv_channel_size, conv_filter_size),
            torch.nn.MaxPool2d(max_pool_size),
            torch.nn.Flatten()
        )

        if not header.config_decomposed["fine_tuning"]:
            for parameter in self.neck.parameters():
                parameter.requires_grad = False

        layer_list = []

        for dataset_entry in config_dataset["attributes"]:
            dataset_name = dataset_entry["name"]
            dataset_labels = dataset_entry["labels"]
            head_hidden_size = header.config_decomposed["head_hidden_size"]

            dataset_labels.remove("")

            layers_hidden = torch.nn.Sequential(torch.nn.Linear(linear_input_size, head_hidden_size), torch.nn.ReLU())
            layer_final = torch.nn.Linear(head_hidden_size, len(dataset_labels))

            layer_dict = torch.nn.ModuleDict({"hidden": layers_hidden, "final": layer_final})
            layer_dict_task = torch.nn.ModuleDict({dataset_name: layer_dict})

            layer_list.append(layer_dict_task)

        self.heads = torch.nn.ModuleList(layer_list)

        return

    def forward(self, input):
        outputs_head = []
        outputs_head_hidden = []
        output_neck = self.neck(input)

        for head in self.heads:
            head_layer_dict = list(head.values())[0]

            output_head_hidden = head_layer_dict["hidden"](output_neck)
            output_head = head_layer_dict["final"](output_head_hidden)

            outputs_head.append(output_head)
            outputs_head_hidden.append(output_head_hidden)

        return (outputs_head, outputs_head_hidden)

    def get_parameters(self):
        if header.config_decomposed["fine_tuning"]:
            return list(self.neck.parameters()) + list(self.heads.parameters())
        else:
            return self.heads.parameters()

class CNNSet(Model):
    def __init__(self, config_dataset, device):
        super().__init__()

        self.model_list = []

        for attribute in config_dataset["attributes"]:
            attribute_labels = attribute["labels"]
            attribute_labels.remove("")

            conv_channel_size = 10
            conv_filter_size = 3
            max_pool_size = 2
            input_size = header.config_decomposed["model_input_height"]
            conv_output_size = (input_size - (conv_filter_size - 1) - (conv_filter_size - 1)) // max_pool_size
            head_hidden_size = header.config_decomposed["head_hidden_size"]
            linear_input_size = conv_output_size * conv_output_size * conv_channel_size
            output_size = len(attribute_labels)

            model = torch.nn.Sequential(
                torch.nn.Conv2d(header.config_decomposed["model_input_channels"], conv_channel_size, conv_filter_size),
                torch.nn.Conv2d(conv_channel_size, conv_channel_size, conv_filter_size),
                torch.nn.MaxPool2d(max_pool_size),
                torch.nn.Flatten(),
                torch.nn.Linear(linear_input_size, head_hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(head_hidden_size, output_size)
            )

            self.model_list.append(model)

        self.model_list = torch.nn.ModuleList(self.model_list)

        return

    def forward(self, input):
        outputs = []

        for model in self.model_list:
            outputs.append(model(input))

        return (outputs, None)

    def get_parameters(self):
        parameters = []

        for model in self.model_list:
            parameters.append(model.parameters())

        return parameters

class DCR(Model):
    def __init__(self, config_dataset, device):
        super().__init__()

        labels_attribute = utility.getLabelsAttribute(config_dataset)
        labels_original = utility.getLabelsOriginal(config_dataset)
        labels_categories = []

        for attribute_name in labels_attribute.keys():
            labels_categories += labels_attribute[attribute_name]

        input_size = header.config_decomposed["model_input_height"] * header.config_decomposed["model_input_width"] * header.config_decomposed["model_input_channels"]
        hidden_size = header.config_decomposed["head_hidden_size"] * len(config_dataset["attributes"])
        output_size = len(labels_categories)
        embedding_size = header.config_reference["model_embedding_size"]

        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU()
        )
        self.concept_embedding = torch_explain.nn.ConceptEmbedding(hidden_size, output_size, embedding_size)
        self.head = torch_explain.nn.concepts.ConceptReasoningLayer(embedding_size, len(labels_original))

    def forward(self, input):
        features = self.net(input)
        concept_embedding, output_neck = self.concept_embedding(features)
        output_head = self.head(concept_embedding, output_neck)

        return (output_neck, output_head)

    def get_parameters(self):
        return list(self.net.parameters()) + list(self.concept_embedding.parameters()) + list(self.head.parameters())

class MLPSet(Model):
    def __init__(self, config_dataset, device):
        super().__init__()

        self.model_list = []

        for attribute in config_dataset["attributes"]:
            attribute_labels = attribute["labels"]
            attribute_labels.remove("")

            input_size = header.config_decomposed["model_input_height"] * header.config_decomposed["model_input_width"] * header.config_decomposed["model_input_channels"]
            hidden_size = header.config_decomposed["head_hidden_size"]
            output_size = len(attribute_labels)

            model = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(input_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, output_size)
            )

            self.model_list.append(model)

        self.model_list = torch.nn.ModuleList(self.model_list)

        return

    def forward(self, input):
        outputs = []

        for model in self.model_list:
            outputs.append(model(input))

        return (outputs, None)

    def get_parameters(self):
        parameters = []

        for model in self.model_list:
            parameters.append(model.parameters())

        return parameters

class MLPSet3(Model):
    def __init__(self, config_dataset, device):
        super().__init__()

        self.model_list = []

        for attribute in config_dataset["attributes"]:
            attribute_labels = attribute["labels"]
            attribute_labels.remove("")

            input_size = header.config_decomposed["model_input_height"] * header.config_decomposed["model_input_width"] * header.config_decomposed["model_input_channels"]
            hidden_size = header.config_decomposed["head_hidden_size"]
            output_size = len(attribute_labels)

            model = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(input_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, output_size)
            )

            self.model_list.append(model)

        self.model_list = torch.nn.ModuleList(self.model_list)

        return

    def forward(self, input):
        outputs = []

        for model in self.model_list:
            outputs.append(model(input))

        return (outputs, None)

    def get_parameters(self):
        parameters = []

        for model in self.model_list:
            parameters.append(model.parameters())

        return parameters

class RelationNN(Model):
    def __init__(self, config_dataset, device):
        super().__init__()

        input_size = 0
        head_hidden_size = header.config_decomposed["head_hidden_size"]
        output_size = len(config_dataset["mappings"])

        for attribute in config_dataset["attributes"]:
            attribute_labels = attribute["labels"]

            if "" in attribute_labels:
                attribute_labels.remove("")

            input_size += len(attribute_labels)

        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_size, head_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(head_hidden_size, output_size)
        )

        return

    def forward(self, inputs):
        return self.model(torch.cat(inputs, 1))

    def get_parameters(self):
        return self.model.parameters()

class ResNet152(Model):
    def __init__(self, config_dataset, device):
        super().__init__()

        self.net = torchvision.models.resnet152(weights = header.config_baseline["model_pretrained_weights"])

        if not header.config_baseline["fine_tuning"]:
            for parameter in self.net.parameters():
                parameter.requires_grad = False

        class_count = len(config_dataset["mappings"].keys())
        net_fc_in_features = self.net.fc.in_features
        self.net.fc = torch.nn.Linear(net_fc_in_features, class_count)

        return

    def forward(self, input):
        return self.net(input)

    def get_parameters(self):
        if header.config_baseline["fine_tuning"]:
            return self.net.parameters()
        else:
            return self.net.fc.parameters()

class ResNet152MTL(Model):
    def __init__(self, config_dataset, device):
        super().__init__()

        self.net = torchvision.models.resnet152(weights = header.config_decomposed["model_pretrained_weights"])

        if not header.config_decomposed["fine_tuning"]:
            for parameter in self.net.parameters():
                parameter.requires_grad = False

        net_fc_in_features = self.net.fc.in_features
        self.net.fc = torch.nn.Identity()

        layer_list = []

        for dataset_entry in config_dataset["attributes"]:
            dataset_name = dataset_entry["name"]
            dataset_labels = dataset_entry["labels"]
            head_hidden_size = header.config_decomposed["head_hidden_size"]

            dataset_labels.remove("")

            layers_hidden = torch.nn.Sequential(torch.nn.Linear(net_fc_in_features, head_hidden_size), torch.nn.ReLU())
            layer_final = torch.nn.Linear(head_hidden_size, len(dataset_labels))

            layer_dict = torch.nn.ModuleDict({"hidden": layers_hidden, "final": layer_final})
            layer_dict_task = torch.nn.ModuleDict({dataset_name: layer_dict})

            layer_list.append(layer_dict_task)

        self.net.heads_mtl = torch.nn.ModuleList(layer_list)

        return

    def forward(self, input):
        outputs_head = []
        outputs_head_hidden = []
        output_neck = self.net(input)

        for head in self.net.heads_mtl:
            head_layer_dict = list(head.values())[0]

            output_head_hidden = head_layer_dict["hidden"](output_neck)
            output_head = head_layer_dict["final"](output_head_hidden)

            outputs_head.append(output_head)
            outputs_head_hidden.append(output_head_hidden)

        return (outputs_head, outputs_head_hidden)

    def get_parameters(self):
        if header.config_decomposed["fine_tuning"]:
            return self.net.parameters()
        else:
            return self.net.heads_mtl.parameters()

class ViTB32(Model):
    def __init__(self, config_dataset, device):
        super().__init__()

        self.net = torchvision.models.vit_b_32(weights = header.config_baseline["model_pretrained_weights"])

        if not header.config_baseline["fine_tuning"]:
            for parameter in self.net.parameters():
                parameter.requires_grad = False

        for parameter in self.net.heads.parameters():
            parameter.requires_grad = True

        class_count = len(config_dataset["mappings"].keys())
        net_heads_head_in_features = self.net.heads.head.in_features
        self.net.heads.head = torch.nn.Linear(net_heads_head_in_features, class_count)

        return

    def forward(self, input):
        return self.net(input)

    def get_parameters(self):
        if header.config_baseline["fine_tuning"]:
            return self.net.parameters()
        else:
            return self.net.heads.parameters()

class ViTB32MTL(Model):
    def __init__(self, config_dataset, device):
        super().__init__()

        self.net = torchvision.models.vit_b_32(weights = header.config_decomposed["model_pretrained_weights"])

        if not header.config_decomposed["fine_tuning"]:
            for parameter in self.net.parameters():
                parameter.requires_grad = False

        for parameter in self.net.heads.parameters():
            parameter.requires_grad = True

        net_heads_head_in_features = self.net.heads.head.in_features
        self.net.heads.head = torch.nn.Identity()

        layer_list = []

        for dataset_entry in config_dataset["attributes"]:
            dataset_name = dataset_entry["name"]
            dataset_labels = dataset_entry["labels"]
            head_hidden_size = header.config_decomposed["head_hidden_size"]

            dataset_labels.remove("")

            layers_hidden = torch.nn.Sequential(torch.nn.Linear(net_heads_head_in_features, head_hidden_size), torch.nn.ReLU())
            layer_final = torch.nn.Linear(head_hidden_size, len(dataset_labels))

            layer_dict = torch.nn.ModuleDict({"hidden": layers_hidden, "final": layer_final})
            layer_dict_task = torch.nn.ModuleDict({dataset_name: layer_dict})

            layer_list.append(layer_dict_task)

        self.net.heads_mtl = torch.nn.ModuleList(layer_list)

        return

    def forward(self, input):
        outputs_head = []
        outputs_head_hidden = []
        output_neck = self.net(input)

        for head in self.net.heads_mtl:
            head_layer_dict = list(head.values())[0]

            output_head_hidden = head_layer_dict["hidden"](output_neck)
            output_head = head_layer_dict["final"](output_head_hidden)

            outputs_head.append(output_head)
            outputs_head_hidden.append(output_head_hidden)

        return (outputs_head, outputs_head_hidden)

    def get_parameters(self):
        if header.config_decomposed["fine_tuning"]:
            return self.net.parameters()
        else:
            return self.net.heads_mtl.parameters()

def createModelBaseline(device):
    file_config_dataset = open(header.file_path_dataset_config, "r")
    config_dataset = json.load(file_config_dataset)
    file_config_dataset.close()

    if header.config_baseline["model"] == type.ModelBaseline.resnet152.name:
        header.config_baseline["model_pretrained_weights"] = "IMAGENET1K_V2"
        logger.log_trace("Model pretrained weights: \"" + header.config_baseline["model_pretrained_weights"] + "\".")
        return ResNet152(config_dataset, device)
    elif header.config_baseline["model"] == type.ModelBaseline.vit_b_32.name:
        header.config_baseline["model_pretrained_weights"] = "IMAGENET1K_V1"
        logger.log_trace("Model pretrained weights: \"" + header.config_baseline["model_pretrained_weights"] + "\".")
        return ViTB32(config_dataset, device)
    else:
        logger.log_fatal("Unknown baseline network model \"" + header.config_baseline["model"] + "\".")
        exit(-1)

def createModelDecomposed(device):
    file_config_dataset = open(header.file_path_dataset_config, "r")
    config_dataset = json.load(file_config_dataset)
    file_config_dataset.close()

    if header.config_decomposed["model"] == type.ModelDecomposed.cnn_mtl.name:
        return CNNMTL(config_dataset, device)
    elif header.config_decomposed["model"] == type.ModelDecomposed.cnn_set.name:
        return CNNSet(config_dataset, device)
    elif header.config_decomposed["model"] == type.ModelDecomposed.mlp_set.name:
        return MLPSet(config_dataset, device)
    elif header.config_decomposed["model"] == type.ModelDecomposed.mlp_set_3.name:
        return MLPSet3(config_dataset, device)
    elif header.config_decomposed["model"] == type.ModelDecomposed.resnet152_mtl.name:
        header.config_decomposed["model_pretrained_weights"] = "IMAGENET1K_V2"
        logger.log_trace("Model pretrained weights: \"" + header.config_decomposed["model_pretrained_weights"] + "\".")
        return ResNet152MTL(config_dataset, device)
    elif header.config_decomposed["model"] == type.ModelDecomposed.vit_b_32_mtl.name:
        header.config_decomposed["model_pretrained_weights"] = "IMAGENET1K_V1"
        logger.log_trace("Model pretrained weights: \"" + header.config_decomposed["model_pretrained_weights"] + "\".")
        return ViTB32MTL(config_dataset, device)
    else:
        logger.log_fatal("Unknown decomposed network model \"" + header.config_decomposed["model"] + "\".")
        exit(-1)

def createModelReference(device):
    file_config_dataset = open(header.file_path_dataset_config, "r")
    config_dataset = json.load(file_config_dataset)
    file_config_dataset.close()

    if header.config_reference["model"] == type.ModelReference.cbm.name:
        return CBM(config_dataset, device)
    elif header.config_reference["model"] == type.ModelReference.cbm_cat.name:
        return CBMCat(config_dataset, device)
    elif header.config_reference["model"] == type.ModelReference.cem.name:
        return CEM(config_dataset, device)
    elif header.config_reference["model"] == type.ModelReference.dcr.name:
        return DCR(config_dataset, device)
    else:
        logger.log_fatal("Unknown reference network model \"" + header.config_reference["model"] + "\".")
        exit(-1)
