import torch
import torch.nn as nn
import numpy as np
import time
from neat.graphs import feed_forward_layers
from . import visualize


class TorchFeedForwardNetwork(nn.Module):
    def __init__(self, inputs, outputs, node_evals):
        super().__init__()
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = nn.ModuleList(node_evals)
        self.values = dict()

    def forward(self, inputs):
        start = time.perf_counter()
        #self.values = dict(
        #   (key, torch.tensor(np.zeros(len(inputs)).reshape((len(inputs), 1)).astype(np.float32))) for key in
        #    self.input_nodes + self.output_nodes)
        self.values = dict()
        end = time.perf_counter()
        #print(f"Conv, Init dict: {end-start}")
        start = time.perf_counter()
        if len(self.input_nodes) != len(inputs[0]):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        if not torch.is_tensor(inputs):
            inputs = torch.from_numpy(inputs).float()

        end = time.perf_counter()
        #print(f"Conv, Error prevention: {end - start}")
        start = time.perf_counter()
        for i, k in enumerate(self.input_nodes):
            self.values[k] = (inputs[:, i])[:, None]
        end = time.perf_counter()
        #print(f"Conv, Init inputs: {end - start}")
        start = time.perf_counter()
        for node in self.node_evals:
            node_inputs = []
            for i, w in node.links:
                node_inputs.append(self.values[i])
            self.values[node.key] = node.forward(torch.cat(node_inputs, dim=1))
        end = time.perf_counter()
        #print(f"Conv, calc values: {end - start}")
        start = time.perf_counter()
        result = torch.cat([self.values[i] for i in self.output_nodes], dim=1)
        end = time.perf_counter()
        #print(f"Conv, filter outputs: {end - start}")
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
            for node in layer:
                inputs = []
                node_expr = []  # currently unused
                for conn_key in connections:
                    inode, onode = conn_key
                    if onode == node:
                        cg = genome.connections[conn_key]
                        inputs.append((inode, cg.weight))
                        node_expr.append("v[{}] * {:.7e}".format(inode, cg.weight))

                ng = genome.nodes[node]
                node_evals.append(NNNode(node, ng.bias, inputs))
        return TorchFeedForwardNetwork(config.input_keys,
                                       config.output_keys, node_evals)


class NNNode(nn.Module):
    def __init__(self, node, bias, inputs):
        super().__init__()
        self.key = node
        self.node = NNNode.init_node(inputs, bias)
        self.bias = bias
        self.links = inputs

    @staticmethod
    def init_node(links, bias):
        linear_layer = nn.Linear(len(links), 1)
        state_dict = linear_layer.state_dict()
        # state_dict['weight'] = torch.from_numpy(np.array([[i[1] for i in links]]))
        # state_dict['bias'] = torch.from_numpy(np.array([bias]))
        linear_layer.load_state_dict(state_dict)
        return linear_layer

    def forward(self, inputs):
        return nn.Sigmoid()(self.node(inputs))


class TorchFeatureExtractionNetwork(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, inputs):
        if not torch.is_tensor(inputs):
            inputs = torch.from_numpy(inputs).float()

        x = inputs
        for layer in self.layers:
            x = layer(x)
        return torch.flatten(x, 1)

    @staticmethod
    def create(inputs, layer_info):
        """ Receives the list of inputs and list of layers """
        layers = []
        previous_nodes = inputs
        for layer in layer_info:
            layers.append(_create_layer(previous_nodes, layer))
            if "nodes" in layer:
                previous_nodes = layer["nodes"]
        return TorchFeatureExtractionNetwork(layers)


def _create_layer(num_inputs, layer):
    if layer["type"] == "conv":
        return nn.Conv2d(num_inputs, layer["nodes"], layer["kernel_size"])
    elif layer["type"] == "pool":
        return nn.MaxPool2d(layer["kernel_size"])
    elif layer["type"] == "relu":
        return nn.ReLU()

class TorchCNN(nn.Module):
    def __init__(self, fe_network, cl_network):
        super().__init__()
        self.feature_extraction = fe_network
        self.classification = cl_network

    def forward(self, inputs):
        if not torch.is_tensor(inputs):
            inputs = torch.from_numpy(inputs).float()

        x = inputs
        x = self.feature_extraction.forward(x)
        x = self.classification.forward(x)
        return x

    @staticmethod
    def create_M1(genome, config):
        fe_network = TorchFeatureExtractionNetwork.create(config.genome_config.image_channels, config.genome_config.feature_extraction_layers)
        cl_network = TorchFeedForwardNetwork.create(genome.classification, config.genome_config.classification_genome_config)
        return TorchCNN(fe_network, cl_network)