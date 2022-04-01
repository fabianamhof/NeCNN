import copy

import torch
import torch.nn as nn
import numpy as np
import time
from neat.graphs import feed_forward_layers
from . import visualize

from torch.nn.utils import prune


class TorchFeedForwardNetwork(nn.Module):
    def __init__(self, inputs, outputs, layers):
        super().__init__()
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.layers = nn.ModuleList(layers)

    def forward(self, inputs):
        # start = time.perf_counter()
        values = dict()
        # end = time.perf_counter()
        # print(f"Conv, Init dict: {end-start}")
        # start = time.perf_counter()
        if len(self.input_nodes) != len(inputs[0]):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        if not torch.is_tensor(inputs):
            inputs = torch.from_numpy(inputs).float()

        # end = time.perf_counter()
        # print(f"Conv, Error prevention: {end - start}")
        # start = time.perf_counter()
        for i, k in enumerate(self.input_nodes):
            values[k] = (inputs[:, i])[:, None]
        # end = time.perf_counter()
        # print(f"Conv, Init inputs: {end - start}")
        # start = time.perf_counter()
        for layer in self.layers:
            #start = time.perf_counter()
            node_inputs = [values[i] for i in layer.inputs]
            #end = time.perf_counter()
            #print(f"Conv, calc inputs: {end - start}")
            #start = time.perf_counter()
            output = layer.forward(torch.cat(node_inputs, dim=1))
            #end = time.perf_counter()
            #print(f"Conv, Forward: {end - start}")
            ##start = time.perf_counter()
            for o, onode in enumerate(layer.nodes):
                values[onode] = output[:, o, None]
            #end = time.perf_counter()
            #print(f"Conv, Write back: {end - start}")

        # end = time.perf_counter()
        # print(f"Conv, calc values: {end - start}")
        # start = time.perf_counter()
        result = torch.cat([values[i] if i in values else torch.zeros((len(inputs), 1)) for i in self.output_nodes], dim=1)
        # end = time.perf_counter()
        # print(f"Conv, filter outputs: {end - start}")
        return result

    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns its phenotype (a FeedForwardNetwork). """

        # Gather expressed connections.
        connections = [cg.key for cg in genome.connections.values() if cg.enabled]

        layers = feed_forward_layers(config.input_keys,
                                     config.output_keys, connections)
        node_evals = []
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
            node_evals.append(NNLayer(nodes, biases, inputs))
        return TorchFeedForwardNetwork(config.input_keys,
                                       config.output_keys, node_evals)


class NNLayer(nn.Module):
    def __init__(self, nodes, bias, inputs):
        super().__init__()
        self.nodes = nodes
        self.bias = bias
        self.links = inputs
        self.layer = self.init_layer()


    def init_layer(self):
        self.inputs = self._get_inputs(self.links)
        layer = nn.Linear(len(self.inputs), len(self.nodes))
        return self._set_weights(layer)

    def _set_weights(self, layer):
        pruning_mask = torch.ones_like(layer.weight)
        for i, inode in enumerate(self.inputs):
            for o, onode in enumerate(self.nodes):
                weight = self._get_weight(inode, onode)
                if weight == 0.0:
                    pruning_mask[o, i] = 0
                with torch.no_grad():
                    layer.weight[o, i] = weight
        with torch.no_grad():
            layer.bias = nn.parameter.Parameter(torch.tensor(self.bias))
        return prune.custom_from_mask(layer, "weight", mask=pruning_mask)


    def _get_weight(self, inode, onode):
        for o, i, weight in self.links:
            if o == onode and i == inode:
                return weight
        return 0.0

    @staticmethod
    def _get_inputs(links):
        inodes = list()
        for (node, inode, weight) in links:
            if inode not in inodes:
                inodes.append(inode)
        return inodes

    def forward(self, inputs):
        return nn.ReLU()(self.layer(inputs))

def create_CNN(genome, config):
    model = config.feature_extraction_model
    model_copy = copy.deepcopy(model)
    classifier = TorchFeedForwardNetwork.create(genome.classification, config.classification_genome_config)
    model_copy.classifier = classifier
    return model_copy
