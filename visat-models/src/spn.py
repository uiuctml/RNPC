import abc
import itertools
import logger
import numpy
import os
import torch

class SPNLearningRateScheduler:
    def __init__(self, optimizer, factor = 0.8, patience = 2, threshold = 1e-4, cooldown = 2, min_learning_rate = 1e-6):
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.factor = factor
        self.metrics = None
        self.min_learning_rate = min_learning_rate
        self.optimizer = optimizer
        self.patience = patience
        self.patience_counter = 0
        self.threshold = threshold

        return

    @abc.abstractmethod
    def step(self, metrics):
        pass

class LikelihoodSPNLearningRateScheduler(SPNLearningRateScheduler):
    def __init__(self, optimizer, factor = 0.8):
        super().__init__(optimizer, factor)

        return

    def step(self, metrics):
        if self.metrics is not None:
            if metrics < self.metrics:
                self.optimizer.learning_rate *= self.factor
                logger.log_info("Reducing SPN learning rate to " + "{:e}".format(self.optimizer.learning_rate) + "...")

        self.metrics = metrics

class LossSPNLearningRateScheduler(SPNLearningRateScheduler):
    def __init__(self, optimizer, factor = 0.8, patience = 2, threshold = 1e-4, cooldown = 2, min_learning_rate = 1e-6):
        super().__init__(optimizer, factor, patience, threshold, cooldown, min_learning_rate)

        return

    def step(self, metrics):
        if self.metrics is None:
            self.metrics = metrics
            return

        if self.optimizer.learning_rate < self.min_learning_rate:
            self.metrics = metrics
            return

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.metrics = metrics
            return

        if abs(self.metrics - metrics) < self.threshold:
            if self.patience_counter < self.patience:
                self.patience_counter += 1
                self.metrics = metrics
                return

            self.optimizer.learning_rate *= self.factor
            logger.log_info("Reducing SPN learning rate to " + "{:e}".format(self.optimizer.learning_rate) + "...")

            self.cooldown_counter = self.cooldown
            self.patience_counter = 0
        else:
            self.patience_counter = 0

        self.metrics = metrics

        return

class SPNOptimizer:
    def __init__(self, spn_joint, spn_marginal, device = torch.device("cuda"), learning_rate = 1e-1, prior_factor = 1e2, projection_epsilon = 1e-2, growth_threshold = 1, marginal_probabilities_counted = None):
        self.device = device
        self.growth_threshold = growth_threshold
        self.learning_rate = learning_rate
        self.marginal_probabilities_counted = marginal_probabilities_counted
        self.smoothing_epsilon = torch.finfo(torch.float).eps
        self.prior_factor = prior_factor
        self.projection_epsilon = projection_epsilon
        self.spn_joint = spn_joint
        self.spn_marginal = spn_marginal
        self.weights_prior = None

        return

    def set_weights_prior(self, weights_prior):
        self.weights_prior = weights_prior

        for i in range(len(self.weights_prior)):
            self.weights_prior[i] *= self.prior_factor

        return

    @abc.abstractmethod
    def step(self, matrix_a = None, matrix_b = None, matrix_c = None, labels_original = None):
        pass

class CCCPDiscriminativeSPNOptimizer(SPNOptimizer):
    def __init__(self, spn_joint, spn_marginal, device = torch.device("cuda"), learning_rate = 1e-1, prior_factor = 1e2, projection_epsilon = 1e-2, growth_threshold = 1, marginal_probabilities_counted = None):
        super().__init__(spn_joint, spn_marginal, device, learning_rate, prior_factor, projection_epsilon, growth_threshold, marginal_probabilities_counted)

        return

    def step(self, matrix_a = None, matrix_b = None, matrix_c = None, labels_original = None):
        self.spn_joint.reuse_forward = False
        self.spn_marginal.reuse_forward = False

        for sum_node in self.spn_joint.sum_nodes:
            weight_normalization_sum_node = 0

            # TODO Optimization: unroll this for loop (compute all weight updates at once)
            for (i, child) in enumerate(sum_node.children):
                # Compute weight updates in log space
                weight_updates = torch.exp(sum_node.value_backward + child.value_forward - self.spn_joint.root_node.value_forward - torch.log(self.marginal_probabilities_counted))
                weight_updates = weight_updates.reshape(matrix_a.shape) # number of original labels x product of category size of all attributes
                weight_updates = weight_updates.t()   # product of category size of all attributes x number of original labels
                weight_updates = torch.index_select(weight_updates, 1, labels_original)   # product of category size of all attributes x batch size
                weight_updates *= matrix_b # product of category size of all attributes x batch size
                weight_updates = torch.sum(weight_updates, 0) # 1 x batch size
                weight_updates = torch.sum(weight_updates)  # 1 x 1
                sum_node.weights[i] *= weight_updates

            # Local weight normalization with Laplace smoothing
            for weight_sum_node in sum_node.weights:
                weight_normalization_sum_node += weight_sum_node + self.smoothing_epsilon

            sum_node.weights = (sum_node.weights + self.smoothing_epsilon) / weight_normalization_sum_node

        self.spn_marginal.set_weights(self.spn_joint.get_weights())

        return

class CCCPGenerativeSPNOptimizer(SPNOptimizer):
    def __init__(self, spn_joint, spn_marginal, device = torch.device("cuda"), learning_rate = 1e-1, prior_factor = 1e2, projection_epsilon = 1e-2, growth_threshold = 1, marginal_probabilities_counted = None):
        super().__init__(spn_joint, spn_marginal, device, learning_rate, prior_factor, projection_epsilon, growth_threshold, marginal_probabilities_counted)

        return

    def step(self, matrix_a = None, matrix_b = None, matrix_c = None, labels_original = None):
        self.spn_joint.reuse_forward = False
        self.spn_marginal.reuse_forward = False

        for sum_node in self.spn_joint.sum_nodes:
            weight_normalization_sum_node = 0

            # TODO Optimization: unroll this for loop (compute all weight updates at once)
            for (i, child) in enumerate(sum_node.children):
                # Compute weight updates in log space
                weight_updates = torch.exp(sum_node.value_backward + child.value_forward - self.spn_joint.root_node.value_forward)
                weight_updates = torch.sum(weight_updates, 0)
                sum_node.weights[i] *= weight_updates

            # Local weight normalization with Laplace smoothing
            for weight_sum_node in sum_node.weights:
                weight_normalization_sum_node += weight_sum_node + self.smoothing_epsilon

            sum_node.weights = (sum_node.weights + self.smoothing_epsilon) / weight_normalization_sum_node

        self.spn_marginal.set_weights(self.spn_joint.get_weights())

        return

class EBWDiscriminativeSPNOptimizer(SPNOptimizer):
    def __init__(self, spn_joint, spn_marginal, device = torch.device("cuda"), learning_rate = 1e-1, prior_factor = 1e2, projection_epsilon = 1e-2, growth_threshold = 1, marginal_probabilities_counted = None):
        super().__init__(spn_joint, spn_marginal, device, learning_rate, prior_factor, projection_epsilon, growth_threshold, marginal_probabilities_counted)

        return

    def step(self, matrix_a = None, matrix_b = None, matrix_c = None, labels_original = None):
        self.spn_joint.reuse_forward = False
        self.spn_marginal.reuse_forward = False

        for (sum_node_joint, sum_node_marginal) in zip(self.spn_joint.sum_nodes, self.spn_marginal.sum_nodes):
            weight_normalization_sum_node = 0

            # TODO Optimization: unroll this for loop (compute all weight updates at once)
            for (i, (child_joint, child_marginal)) in enumerate(zip(sum_node_joint.children, sum_node_marginal.children)):
                # Compute weight updates in log space
                weight_updates_joint = torch.exp(sum_node_joint.value_backward + child_joint.value_forward - self.spn_joint.root_node.value_forward)
                weight_updates_marginal = torch.exp(sum_node_marginal.value_backward + child_marginal.value_forward - self.spn_marginal.root_node.value_forward)
                weight_updates = weight_updates_joint - weight_updates_marginal
                weight_updates *= torch.exp(self.spn_joint.root_node.value_forward)
                weight_updates += self.growth_threshold
                weight_updates = weight_updates.reshape(matrix_a.shape) # number of original labels x product of category size of all attributes
                weight_updates = weight_updates.t()   # product of category size of all attributes x number of original labels
                weight_updates = torch.index_select(weight_updates, 1, labels_original)   # product of category size of all attributes x batch size
                weight_updates *= matrix_b # product of category size of all attributes x batch size
                weight_updates = torch.sum(weight_updates, 0) # 1 x batch size
                weight_updates = torch.mean(weight_updates, 0)  # 1 x 1
                sum_node_joint.weights[i] *= weight_updates

            # Local weight normalization with Laplace smoothing
            for weight_sum_node in sum_node_joint.weights:
                weight_normalization_sum_node += weight_sum_node + self.smoothing_epsilon

            sum_node_joint.weights = (sum_node_joint.weights + self.smoothing_epsilon) / weight_normalization_sum_node

        self.spn_marginal.set_weights(self.spn_joint.get_weights())

        return

class PGDDiscriminativeSPNOptimizer(SPNOptimizer):
    def __init__(self, spn_joint, spn_marginal, device = torch.device("cuda"), learning_rate = 1e-1, prior_factor = 1e2, projection_epsilon = 1e-2, growth_threshold = 1, marginal_probabilities_counted = None):
        super().__init__(spn_joint, spn_marginal, device, learning_rate, prior_factor, projection_epsilon, growth_threshold, marginal_probabilities_counted)

        return

    def step(self, matrix_a = None, matrix_b = None, matrix_c = None, labels_original = None):
        self.spn_joint.reuse_forward = False
        self.spn_marginal.reuse_forward = False

        matrix_c_transposed = matrix_c.t()  # number of original labels x batch size

        for (sum_node_joint, sum_node_marginal, weights_prior) in zip(self.spn_joint.sum_nodes, self.spn_marginal.sum_nodes, self.weights_prior):
            # TODO Optimization: unroll this for loop (compute all weight updates at once)
            for (i, (child_joint, child_marginal)) in enumerate(zip(sum_node_joint.children, sum_node_marginal.children)):
                # Compute weight updates in log space
                weight_updates_joint = torch.exp(sum_node_joint.value_backward + child_joint.value_forward - self.spn_joint.root_node.value_forward)
                weight_updates_marginal = torch.exp(sum_node_marginal.value_backward + child_marginal.value_forward - self.spn_marginal.root_node.value_forward)

                # Set gradients corresponding to zero root node forward values to zero
                mask_1_joint = ((sum_node_joint.value_backward + child_joint.value_forward) == -float("inf"))
                mask_2_joint = (self.spn_joint.root_node.value_forward == -float("inf"))
                mask_joint = mask_1_joint & mask_2_joint
                mask_1_marginal = ((sum_node_marginal.value_backward + child_marginal.value_forward) == -float("inf"))
                mask_2_marginal = (self.spn_marginal.root_node.value_forward == -float("inf"))
                mask_marginal = mask_1_marginal & mask_2_marginal
                weight_updates_joint[mask_joint] = 0
                weight_updates_marginal[mask_marginal] = 0

                weight_updates = weight_updates_joint - weight_updates_marginal
                weight_updates = weight_updates.reshape(matrix_a.shape) # number of original labels x product of category size of all attributes
                weight_updates *= matrix_a  # number of original labels x product of category size of all attributes
                weight_updates = weight_updates.t()   # product of category size of all attributes x number of original labels
                weight_updates = torch.index_select(weight_updates, 1, labels_original)   # product of category size of all attributes x batch size
                weight_updates *= matrix_b # product of category size of all attributes x batch size
                weight_updates = torch.sum(weight_updates, 0) # 1 x batch size
                weight_updates /=  matrix_c_transposed[labels_original, torch.arange(matrix_c_transposed.shape[1])]   # 1 x batch size

                # Average weight updates with Dirichlet prior
                weight_updates_count = weight_updates.shape[0]
                weight_updates = torch.sum(weight_updates, 0, keepdim = True)
                weight_updates += (weights_prior[i] - 1) / sum_node_joint.weights[i]
                weight_updates /= weight_updates_count

                sum_node_joint.weights[i] += self.learning_rate * weight_updates

                if sum_node_joint.weights[i] <= 0:
                    sum_node_joint.weights[i] = self.projection_epsilon

        self.spn_marginal.set_weights(self.spn_joint.get_weights())

        return

class PGDGenerativeSPNOptimizer(SPNOptimizer):
    def __init__(self, spn_joint, spn_marginal, device = torch.device("cuda"), learning_rate = 1e-1, prior_factor = 1e2, projection_epsilon = 1e-2, growth_threshold = 1, marginal_probabilities_counted = None):
        super().__init__(spn_joint, spn_marginal, device, learning_rate, prior_factor, projection_epsilon, growth_threshold, marginal_probabilities_counted)

        return

    def step(self, matrix_a = None, matrix_b = None, matrix_c = None, labels_original = None):
        self.spn_joint.reuse_forward = False
        self.spn_marginal.reuse_forward = False

        for (sum_node_joint, sum_node_marginal, weights_prior) in zip(self.spn_joint.sum_nodes, self.spn_marginal.sum_nodes, self.weights_prior):
            # TODO Optimization: unroll this for loop (compute all weight updates at once)
            for (i, (child_joint, child_marginal)) in enumerate(zip(sum_node_joint.children, sum_node_marginal.children)):
                # Compute weight updates in log space
                weight_updates_joint = torch.exp(sum_node_joint.value_backward + child_joint.value_forward - self.spn_joint.root_node.value_forward)
                weight_updates_marginal = torch.exp(sum_node_marginal.value_backward + child_marginal.value_forward - self.spn_marginal.root_node.value_forward)
                weight_updates = weight_updates_joint - weight_updates_marginal

                # Average weight updates with Dirichlet prior
                weight_updates_count = weight_updates.shape[0]
                weight_updates = torch.sum(weight_updates, 0, keepdim = True)
                weight_updates += (weights_prior[i] - 1) / sum_node_joint.weights[i]
                weight_updates /= weight_updates_count

                sum_node_joint.weights[i] += self.learning_rate * weight_updates

                if sum_node_joint.weights[i] <= 0:
                    sum_node_joint.weights[i] = self.projection_epsilon

        self.spn_marginal.set_weights(self.spn_joint.get_weights())

        return

class Node:
    def __init__(self):
        self.children = []
        self.depth = None
        self.device = None
        self.id = None
        self.parents = []
        self.value_backward = None
        self.value_forward = None

        return

    def backward(self):
        if len(self.parents) == 0:
            logger.log_fatal("Node " + str(self.id) + " has no parents. Quit.")
            exit(-1)

        value_backward_parents_product = []
        value_backward_parents_sum = []
        value_backward_product = None
        value_backward_sum = None
        value_forward_parents_product = []
        weights_parents_sum = []

        for parent in self.parents:
            if isinstance(parent, ProductNode):
                value_backward_parents_product.append(parent.value_backward)
                value_forward_parents_product.append(parent.value_forward)
            elif isinstance(parent, SumNode):
                value_backward_parents_sum.append(parent.value_backward)
                weights_parents_sum.append(parent.weights[parent.weights_index_by_child_id[self.id]])

        if len(value_backward_parents_product) > 0:
            value_backward_parents_product = torch.stack(value_backward_parents_product)

        if len(value_backward_parents_sum) > 0:
            value_backward_parents_sum = torch.stack(value_backward_parents_sum)

        if len(value_forward_parents_product) > 0:
            value_forward_parents_product = torch.stack(value_forward_parents_product)

        if len(weights_parents_sum) > 0:
            weights_parents_sum = torch.stack(weights_parents_sum)
            weights_parents_sum = torch.Tensor(weights_parents_sum).reshape(-1, 1)
            weights_parents_sum = weights_parents_sum.to(self.device)

        if len(value_backward_parents_product) > 0:
            value_backward_product = value_backward_parents_product + value_forward_parents_product - self.value_forward

        if len(value_backward_parents_sum) > 0:
            value_backward_sum = value_backward_parents_sum + torch.log(weights_parents_sum)

        if value_backward_product is not None and value_backward_sum is None:
            self.value_backward = value_backward_product
        elif value_backward_product is None and value_backward_sum is not None:
            self.value_backward = value_backward_sum
        else:
            self.value_backward = torch.stack([value_backward_product, value_backward_sum])

        # Compute backward values in log space
        # Log-Sum-Exp trick: https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        value_backward_max = torch.max(self.value_backward, 0)[0]
        self.value_backward -= value_backward_max
        self.value_backward = torch.exp(self.value_backward)
        self.value_backward = torch.sum(self.value_backward, 0)
        self.value_backward = torch.log(self.value_backward) + value_backward_max

        return

    @abc.abstractmethod
    def forward(self):
        pass

class CategoricalLeafNode(Node):
    def __init__(self):
        super().__init__()

        self.attribute_index = None
        self.category_index = None

        return

    def forward(self):
        return

    def set(self, settings):
        if self.attribute_index < 0 or self.category_index < 0:
            logger.log_fatal("Invalid categorical leaf node. Quit.")
            exit(-1)

        variables = settings[:, self.attribute_index]
        settings = (variables == self.category_index)
        settings_marginal = (variables < 0)

        self.value_forward = torch.logical_or(settings, settings_marginal).float()
        self.value_forward = self.value_forward.to(self.device)

        # Compute forward values in log space
        self.value_forward = torch.log(self.value_forward)

        return

class ProductNode(Node):
    def __init__(self):
        super().__init__()

        return

    def forward(self):
        if len(self.children) == 0:
            logger.log_fatal("Product node " + str(self.id) + " has no children. Quit.")
            exit(-1)

        value_forward_children = []

        for child in self.children:
            value_forward_children.append(child.value_forward)

        # Compute forward values in log space
        value_forward_children = torch.stack(value_forward_children)
        self.value_forward = torch.sum(value_forward_children, 0)

        return

class SumNode(Node):
    def __init__(self):
        super().__init__()

        self.leaf = False
        self.weights = []
        self.weights_index_by_child_id = {}

        return

    def forward(self):
        if len(self.children) == 0:
            logger.log_fatal("Sum node " + str(self.id) + " has no children. Quit.")
            exit(-1)

        value_forward_children = []

        for child in self.children:
            value_forward_children.append(child.value_forward)

        # Compute forward values in log space
        # Log-Sum-Exp trick: https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
        value_forward_children = torch.stack(value_forward_children)
        value_forward_children_max = torch.max(value_forward_children, 0)[0]
        value_forward_children_max[value_forward_children_max == -float("inf")] = 0
        value_forward_children -= value_forward_children_max
        value_forward_children = torch.exp(value_forward_children)
        value_forward_children *= self.weights
        self.value_forward = torch.sum(value_forward_children, 0)
        self.value_forward = torch.log(self.value_forward) + value_forward_children_max

        return

class SPN:
    def __init__(self, device = torch.device("cuda")):
        self.depth = None
        self.device = device
        self.induced_trees = []
        self.leaf_nodes = []
        self.leaf_nodes_dict = {}
        self.nodes = []
        self.product_nodes = []
        self.reuse_backward = False
        self.reuse_forward = False
        self.root_node = None
        self.settings = None
        self.sum_nodes = []
        self.traversal_order_backward = []
        self.traversal_order_forward = []

        return

    def __call__(self, settings):
        self.set_leaf_nodes(settings)

        return self.forward()

    def backward(self):
        if not self.reuse_backward:
            if len(self.traversal_order_backward) == 0:
                logger.log_fatal("Empty tree. Quit.")
                exit(-1)

            if self.root_node is None or self.traversal_order_backward[0].id != self.root_node.id:
                logger.log_fatal("Missing root node. Quit.")
                exit(-1)

            if self.root_node.value_forward is None:
                logger.log_fatal("Missing root node forward value. Quit.")
                exit(-1)

            self.reuse_backward = True

            # Initialize root node backward value in log space
            self.root_node.value_backward = torch.log(torch.ones(self.settings.shape[0]))
            self.root_node.value_backward = self.root_node.value_backward.to(self.device)

            for node in self.traversal_order_backward:
                # Skip root node
                if node.id == self.root_node.id:
                    continue

                node.backward()

        return

    def forward(self):
        if not self.reuse_forward:
            if len(self.traversal_order_forward) == 0:
                logger.log_fatal("Empty tree. Quit.")
                exit(-1)

            if self.root_node is None or self.traversal_order_forward[-1].id != self.root_node.id:
                logger.log_fatal("Missing root node. Quit.")
                exit(-1)

            self.reuse_backward = False
            self.reuse_forward = True

            for node in self.traversal_order_forward:
                node.forward()

        return self.root_node.value_forward

    def gather_induced_trees(self):
        for induced_tree in self.recurse_induced_trees(self.root_node):
            induced_tree[1] = numpy.prod(induced_tree[1])
            self.induced_trees.append(induced_tree)

        return

    def get_weights(self):
        weights = []

        for sum_node in self.sum_nodes:
            weights.append(torch.clone(sum_node.weights))

        return weights

    def load(self, file_path_spn):
        if not os.path.exists(file_path_spn):
            logger.log_fatal("Invalid SPN file path. Quit.")
            exit(-1)

        self.reuse_backward = False
        self.reuse_forward = False

        with open(file_path_spn, "r") as file_spn:
            categorical_leaf_node_id = -1
            reading_nodes = True

            for line in file_spn.readlines():
                line = line.strip()

                if line[0] == "#":
                    line = line.replace("#", "")

                    if line == "NODES":
                        reading_nodes = True
                    elif line == "EDGES":
                        reading_nodes = False

                    continue

                line_list = line.split(",")

                if reading_nodes:
                    node_id = int(line_list[0])
                    node_type = line_list[1]

                    if node_type == "SUM":
                        sum_node = SumNode()
                        sum_node.device = self.device
                        sum_node.id = node_id
                        self.nodes.append(sum_node)
                        self.sum_nodes.append(sum_node)
                    elif node_type == "PRD":
                        product_node = ProductNode()
                        product_node.device = self.device
                        product_node.id = node_id
                        self.nodes.append(product_node)
                        self.product_nodes.append(product_node)
                    elif node_type == "CatNode" or node_type == "CATNODE":
                        categorical_leaf_node_list = []
                        node_attribute_index = int(line_list[2])
                        node_probabilities = line_list[3:]

                        for i in range(0, len(node_probabilities)):
                            node_probabilities[i] = float(node_probabilities[i])

                        if node_attribute_index in self.leaf_nodes_dict.keys():
                            categorical_leaf_node_list = self.leaf_nodes_dict[node_attribute_index]
                        else:
                            for category_index in range(0, len(node_probabilities)):
                                categorical_leaf_node = CategoricalLeafNode()
                                categorical_leaf_node.attribute_index = node_attribute_index
                                categorical_leaf_node.category_index = category_index
                                categorical_leaf_node.device = self.device
                                categorical_leaf_node.id = categorical_leaf_node_id
                                categorical_leaf_node_id -= 1

                                categorical_leaf_node_list.append(categorical_leaf_node)
                                self.leaf_nodes.append(categorical_leaf_node)
                                self.nodes.append(categorical_leaf_node)

                            self.leaf_nodes_dict[node_attribute_index] = categorical_leaf_node_list

                        sum_node = SumNode()
                        sum_node.children = categorical_leaf_node_list
                        sum_node.device = self.device
                        sum_node.id = node_id
                        sum_node.leaf = True
                        sum_node.weights = node_probabilities
                        self.nodes.append(sum_node)
                        self.sum_nodes.append(sum_node)

                        for (i, categorical_leaf_node) in enumerate(categorical_leaf_node_list):
                            sum_node.weights_index_by_child_id[categorical_leaf_node.id] = i
                            categorical_leaf_node.parents.append(sum_node)
                    elif node_type == "CATNODEPRD":
                        node_attribute_index = int(line_list[2])
                        node_category_index = int(line_list[3])

                        categorical_leaf_node = CategoricalLeafNode()
                        categorical_leaf_node.attribute_index = node_attribute_index
                        categorical_leaf_node.category_index = node_category_index
                        categorical_leaf_node.device = self.device
                        categorical_leaf_node.id = node_id

                        self.leaf_nodes.append(categorical_leaf_node)
                        self.nodes.append(categorical_leaf_node)
                else:
                    nodes = []
                    node_id_first = int(line_list[0])
                    node_id_second = int(line_list[1])

                    for node in self.nodes:
                        if node.id == node_id_first or node.id == node_id_second:
                            nodes.append(node)

                    if len(nodes) != 2:
                        logger.log_fatal("Invalid edge.")
                        exit(-1)

                    if len(line_list) >= 3:
                        node_weight = float(line_list[2])

                        if isinstance(nodes[0], SumNode) and not nodes[0].leaf:
                            nodes[0].children.append(nodes[1])
                            nodes[0].weights.append(node_weight)
                            nodes[0].weights_index_by_child_id[nodes[1].id] = len(nodes[0].weights) - 1
                            nodes[1].parents.append(nodes[0])
                        elif isinstance(nodes[1], SumNode) and not nodes[0].leaf:
                            nodes[1].children.append(nodes[0])
                            nodes[1].weights.append(node_weight)
                            nodes[1].weights_index_by_child_id[nodes[0].id] = len(nodes[1].weights) - 1
                            nodes[0].parents.append(nodes[1])
                    else:
                        if isinstance(nodes[0], ProductNode):
                            nodes[0].children.append(nodes[1])
                            nodes[1].parents.append(nodes[0])
                        elif isinstance(nodes[1], ProductNode):
                            nodes[1].children.append(nodes[0])
                            nodes[0].parents.append(nodes[1])

        for sum_node in self.sum_nodes:
            sum_node.weights = torch.Tensor(sum_node.weights).reshape(-1, 1)
            sum_node.weights = sum_node.weights.to(self.device)

        root_nodes = []

        for node in self.nodes:
            if len(node.parents) == 0:
                root_nodes.append(node)

        if len(root_nodes) != 1:
            logger.log_fatal("Invalid SPN.")
            exit(-1)

        self.depth = self.traverse({0: root_nodes}, 0)
        self.root_node = root_nodes[0]
        self.traversal_order_backward = self.topological_sort()
        self.traversal_order_forward = self.traversal_order_backward.copy()
        self.traversal_order_forward.reverse()

        if len(self.traversal_order_backward) != len(self.nodes):
            logger.log_fatal("Invalid SPN backward traversal.")
            exit(-1)

        if len(self.traversal_order_forward) != len(self.nodes):
            logger.log_fatal("Invalid SPN forward traversal.")
            exit(-1)

        return

    def normalize_weights(self, smoothing_epsilon):
        for sum_node in self.sum_nodes:
            value_forward_children = []
            weight_normalization = 0

            for child in sum_node.children:
                value_forward_children.append(child.value_forward)

            value_forward_children = torch.stack(value_forward_children)
            value_forward_children_max = torch.max(value_forward_children, 0)[0]

            for (i, child) in enumerate(sum_node.children):
                weight_normalization += sum_node.weights[i] * torch.exp(child.value_forward - value_forward_children_max) + smoothing_epsilon

            # Local weight normalization with Laplace smoothing
            for (i, child) in enumerate(sum_node.children):
                weight = sum_node.weights[i] * torch.exp(child.value_forward - value_forward_children_max) + smoothing_epsilon
                sum_node.weights[i] = weight / weight_normalization

        return

    def randomize_weights(self):
        self.reuse_backward = False
        self.reuse_forward = False

        weights = self.get_weights()

        for i in range(len(weights)):
            weights[i] = weights[i].uniform_(0, 1)
            weights[i] /= torch.sum(weights[i])

        self.set_weights(weights)

        return

    def recurse_induced_trees(self, node):
        if isinstance(node, SumNode):
            for (i, child) in enumerate(node.children):
                for sub_induced_tree in self.recurse_induced_trees(child):
                    induced_tree = [{node}, [node.weights[i].item()]]
                    induced_tree[0].update(sub_induced_tree[0])
                    induced_tree[1] += sub_induced_tree[1]
                    yield induced_tree
        elif isinstance(node, ProductNode):
            sub_induced_trees_product = []

            for child in node.children:
                sub_induced_trees = []
                for sub_induced_tree in self.recurse_induced_trees(child):
                    sub_induced_trees.append(sub_induced_tree)
                sub_induced_trees_product.append(sub_induced_trees)

            for sub_induced_trees in itertools.product(*sub_induced_trees_product):
                induced_tree = [{node}, []]
                for sub_induced_tree in sub_induced_trees:
                    induced_tree[0].update(sub_induced_tree[0])
                    induced_tree[1] += sub_induced_tree[1]
                yield induced_tree
        elif isinstance(node, CategoricalLeafNode):
            yield [{node}, []]

        return

    def set_leaf_nodes(self, settings):
        self.reuse_backward = False
        self.reuse_forward = False
        self.settings = settings

        for leaf_node in self.leaf_nodes:
            leaf_node.set(self.settings)

        return

    def set_weights(self, weights):
        if len(weights) != len(self.sum_nodes):
            logger.log_fatal("Invalid weights. Quit.")
            exit(-1)

        self.reuse_backward = False
        self.reuse_forward = False

        for (sum_node, sum_node_weights) in zip(self.sum_nodes, weights):
            sum_node.weights = torch.clone(sum_node_weights)

        return

    def topological_sort(self):
        parent_count = {}
        queue = []
        topological_order = []

        for node in self.nodes:
            parent_count[node.id] = len(node.parents)

        queue.append(self.root_node)

        while len(queue) > 0:
            node = queue[0]
            topological_order.append(node)

            for child in node.children:
                parent_count[child.id] -= 1

                if parent_count[child.id] <= 0:
                    queue.append(child)

            queue.pop(0)

        return topological_order

    def traverse(self, layers, depth):
        if depth not in layers.keys():
            return depth - 1

        for node in layers[depth]:
            node.depth = depth

            for child in node.children:
                if depth + 1 not in layers.keys():
                    layers[depth + 1] = []

                layers[depth + 1].append(child)

        depth_next = self.traverse(layers, depth + 1)

        if depth_next > depth:
            depth = depth_next

        return depth
