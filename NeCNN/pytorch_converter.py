import torch
import torch.nn as nn
import numpy as np
import time
from neat.graphs import feed_forward_layers
from . import visualize



class TorchFeedForwardNetwork(torch.jit.ScriptModule):
    def __init__(self, inputs, outputs, node_evals):
        super().__init__()
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = nn.ModuleList(node_evals)
        self.values = dict()

    def forward(self, inputs):
        # start = time.perf_counter()
        self.values = dict()
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
            self.values[k] = (inputs[:, i])[:, None]
        # end = time.perf_counter()
        # print(f"Conv, Init inputs: {end - start}")
        # start = time.perf_counter()
        for node in self.node_evals:
            node_inputs = [self.values[i] for i, w in node.links]
            self.values[node.key] = node.forward(torch.cat(node_inputs, dim=1))
        # end = time.perf_counter()
        # print(f"Conv, calc values: {end - start}")
        # start = time.perf_counter()
        result = torch.cat([self.values[i] if i in self.values else torch.zeros((len(inputs), 1)) for i in self.output_nodes], dim=1)
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


class NNNode(torch.jit.ScriptModule):
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