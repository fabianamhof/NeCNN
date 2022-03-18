import torch
import torch.nn as nn
import numpy as np

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
        self.values = dict((key, torch.tensor(np.zeros(len(inputs)).reshape((len(inputs),1)).astype(np.float32))) for key in self.input_nodes + self.output_nodes)
        if len(self.input_nodes) != len(inputs[0]):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        if not torch.is_tensor(inputs):
            inputs = torch.from_numpy(inputs).float()

        for i, k in enumerate(self.input_nodes):
            self.values[k] = inputs[:, i].reshape((len(inputs),1))

        for node in self.node_evals:
            node_inputs = []
            for i, w in node.links:
                node_inputs.append(self.values[i])
            self.values[node.key] = node.forward(torch.cat(node_inputs, dim=1))

        return torch.cat([self.values[i] for i in self.output_nodes], dim=1)

    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns its phenotype (a FeedForwardNetwork). """

        # Gather expressed connections.
        connections = [cg.key for cg in genome.connections.values() if cg.enabled]

        layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connections)
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
                aggregation_function = config.genome_config.aggregation_function_defs.get(ng.aggregation)
                activation_function = config.genome_config.activation_defs.get(ng.activation)
                node_evals.append(NNNode(node, activation_function, ng.bias, inputs))
        return TorchFeedForwardNetwork(config.genome_config.input_keys, config.genome_config.output_keys, node_evals)


class NNNode(nn.Module):
    def __init__(self, node, activation_function, bias, inputs):
        super().__init__()
        self.key = node
        self.node = NNNode.init_node(inputs, bias)
        self.activation_function = activation_function
        self.bias = bias
        self.links = inputs

    @staticmethod
    def init_node(links, bias):
        linear_layer = nn.Linear(len(links), 1)
        state_dict = linear_layer.state_dict()
        state_dict['weight'] = torch.from_numpy(np.array([[i[1] for i in links]]))
        state_dict['bias'] = torch.from_numpy(np.array([bias]))
        linear_layer.load_state_dict(state_dict)
        return linear_layer

    def forward(self, inputs):
        return nn.Sigmoid()(self.node(inputs))
