import copy

import torch
import torch.nn as nn
from neat.graphs import feed_forward_layers
from .net import Net

from torch.nn.utils import prune


class TorchFeedForwardNetwork(nn.Module):
    """
    A custom pytorch feedforward Network.
    """

    def __init__(self, inputs, outputs, layers, node_mapping):
        """
        Initializes the network.
        :param inputs: Input Nodes
        :param outputs: Output Nodes
        :param layers: List of layers
        :param node_mapping: Maps Node number to entry in list of all nodes. Example: -320 -> 0, 1-> 321,...
        """
        super().__init__()
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_mapping = node_mapping
        self.layers = nn.ModuleList(layers)

    def forward(self, inputs):
        """
        Forward pass through the network
        :param inputs: Input data
        :return: Output of the network
        """
        if len(self.input_nodes) != len(inputs[0]):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs[0])))

        if not torch.is_tensor(inputs):
            inputs = torch.from_numpy(inputs).float()
        # Prepare a tensor that contains the outputs of all nodes
        values = torch.zeros((len(inputs), len(self.node_mapping))).to(inputs.device)
        values[:, 0:len(inputs[0])] = inputs  # First columns are for input nodes
        for layer in self.layers:
            # Get all inputs for the specific layer
            node_inputs = values[:, [self.node_mapping[i] for i in layer.inputs]]
            # Forward inputs
            output = layer.forward(node_inputs)
            # Write outputs back to the tensor
            values[:, [self.node_mapping[i] for i in layer.nodes]] = output

        # Get all outputs and return
        result = values[:, [self.node_mapping[i] for i in self.output_nodes]]
        return result

    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns its phenotype (a FeedForwardNetwork). """

        # Gather expressed connections.
        connections = [cg.key for cg in genome.connections.values() if cg.enabled]

        # Get layers sorted by depth
        layers = feed_forward_layers(config.input_keys,
                                     config.output_keys, connections)

        node_mapping = {}
        # Map input keys to the first elements in array
        for i, key in enumerate(sorted(config.input_keys)):
            node_mapping[key] = i
        # Append all other nodes
        for i, node in enumerate(genome.nodes):
            node_mapping[node] = len(config.input_keys) + i

        layer_evals = []
        # Calculate all nodes, biases and inputs for each layer
        for layer in layers:
            nodes = []
            biases = []
            inputs = []
            for node in layer:
                nodes.append(node)
                if hasattr(genome.nodes[node], "bias"):
                    biases.append(genome.nodes[node].bias)
                for conn_key in connections:
                    inode, onode = conn_key
                    if onode == node:
                        cg = genome.connections[conn_key]
                        if hasattr(cg, "weight"):
                            inputs.append((onode, inode, cg.weight))
                        else:
                            inputs.append((onode, inode))
            layer_evals.append(NNLayer(nodes, biases, inputs))
        return TorchFeedForwardNetwork(config.input_keys,
                                       config.output_keys, layer_evals, node_mapping)


class NNLayer(nn.Module):
    """
    A pytorch linear layer with some additional infos.
    """

    def __init__(self, nodes, bias, links):
        """

        :param nodes: Output nodes of the layer
        :param bias: Biases of the output nodes
        :param links: Connections from previous nodes to the layer
        """
        super().__init__()
        self.nodes = nodes
        self.bias = bias
        self.links = links
        self.layer = self.init_layer()

    def init_layer(self):
        self.inputs = self._get_inputs(self.links)
        layer = nn.Linear(len(self.inputs), len(self.nodes))
        return self._set_weights(layer)

    def _set_weights(self, layer):
        pruning_mask = torch.zeros_like(layer.weight)
        input_mapping = {inode: i for i, inode in enumerate(self.inputs)}
        output_mapping = {onode: o for o, onode in enumerate(self.nodes)}
        if len(self.links[0]) == 3:  # Has weights
            for o, i, weight in self.links:
                with torch.no_grad():
                    layer.weight[output_mapping[o], input_mapping[i]] = weight
                    pruning_mask[output_mapping[o], input_mapping[i]] = 1  # Enable connection in the pruning mask
        else:
            for o, i in self.links:
                with torch.no_grad():
                    pruning_mask[output_mapping[o], input_mapping[i]] = 1
        if len(self.bias) > 0:  # has bias
            with torch.no_grad():
                layer.bias = nn.parameter.Parameter(torch.tensor(self.bias))
        return prune.custom_from_mask(layer, "weight", mask=pruning_mask)

    @staticmethod
    def _get_inputs(links):
        """

        :param links:
        :return: returns unique list of all input nodes
        """
        inodes = list()
        for _, inode, *_ in links:
            if inode not in inodes:
                inodes.append(inode)
        inodes.sort()
        return inodes

    def forward(self, inputs):
        return nn.ReLU()(self.layer(inputs))


def create_CNN(features, classifier):
    """
    Creates CNN by assigning feature extraction and classifier to a pytorch Network.
    Feature extraction will be deepcopied, classifier not.
    :param features: Feature extraction
    :param classifier: Classifier
    :return:
    """
    net = Net()
    net.features = copy.deepcopy(features)
    net.classifier = classifier
    return net
