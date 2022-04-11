import copy

import torch
import torch.nn as nn
import numpy as np
import time
from neat.graphs import feed_forward_layers
from NeCNN import visualize

from torch.nn.utils import prune


class TorchFeedForwardNetwork(nn.Module):
    def __init__(self, inputs, outputs, layers, node_mapping):
        super().__init__()
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_mapping = node_mapping
        self.layers = nn.ModuleList(layers)

    def forward(self, inputs):
        if len(self.input_nodes) != len(inputs[0]):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        if not torch.is_tensor(inputs):
            inputs = torch.from_numpy(inputs).float()

        values = torch.zeros((len(inputs), len(self.node_mapping)))
        values[:, 0:len(inputs[0])] = inputs  # First columns are for input nodes
        for layer in self.layers:
            node_inputs = values[:, [self.node_mapping[i] for i in layer.inputs]]
            output = layer.forward(node_inputs)
            values[:, [self.node_mapping[i] for i in layer.nodes]] = output

        result = values[:, [self.node_mapping[i] for i in self.output_nodes]]
        return result

    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns its phenotype (a FeedForwardNetwork). """

        # Gather expressed connections.
        connections = [cg.key for cg in genome.connections.values() if cg.enabled]

        layers = feed_forward_layers(config.input_keys,
                                     config.output_keys, connections)

        node_mapping = {}
        for i, key in enumerate(sorted(config.input_keys)):
            node_mapping[key] = i
        for i, node in enumerate(genome.nodes):
            node_mapping[node] = i
        layer_evals = []
        for layer in layers:
            nodes = []
            biases = []
            inputs = []
            for node in layer:
                nodes.append(node)
                biases.append(genome.nodes[node].bias)
                for conn_key in connections:
                    inode, onode = conn_key
                    if onode == node:
                        cg = genome.connections[conn_key]
                        inputs.append((onode, inode, cg.weight))
            layer_evals.append(NNLayer(nodes, biases, inputs))
        return TorchFeedForwardNetwork(config.input_keys,
                                       config.output_keys, layer_evals, node_mapping)


class NNLayer(nn.Module):
    def __init__(self, nodes, bias, links):
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
        for o, i, weight in self.links:
            with torch.no_grad():
                # layer.weight[output_mapping[o], input_mapping[i]] = weight
                pruning_mask[output_mapping[o], input_mapping[i]] = 1
        # with torch.no_grad():
        #    layer.bias = nn.parameter.Parameter(torch.tensor(self.bias))
        return prune.custom_from_mask(layer, "weight", mask=pruning_mask)

    @staticmethod
    def _get_inputs(links):
        inodes = list()
        for (node, inode, weight) in links:
            if inode not in inodes:
                inodes.append(inode)
        inodes.sort()
        return inodes

    def forward(self, inputs):
        return nn.ReLU()(self.layer(inputs))


def create_CNN(genome, config):
    model = config.feature_extraction_model
    model_copy = copy.deepcopy(model)
    classifier = TorchFeedForwardNetwork.create(genome.classification, config.classification_genome_config)
    model_copy.classifier = classifier
    return model_copy
