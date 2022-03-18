"""Handles genomes (individuals in the population)."""
from __future__ import division, print_function

from neat.genome import *
from itertools import count
from random import choice, random, shuffle

import sys

from neat.activations import ActivationFunctionSet
from neat.aggregations import AggregationFunctionSet
from neat.config import ConfigParameter, write_pretty_params
from neat.genes import DefaultConnectionGene, DefaultNodeGene
from neat.graphs import creates_cycle


class CnnGenome1Config(DefaultGenomeConfig):
    """Sets up and holds configuration information for the DefaultGenome class."""
    allowed_connectivity = ['unconnected', 'fs_neat_nohidden', 'fs_neat', 'fs_neat_hidden',
                            'full_nodirect', 'full', 'full_direct',
                            'partial_nodirect', 'partial', 'partial_direct']

    def __init__(self, params):
        super().__init__(params)
        self._params += [ConfigParameter('test', int)]
        for p in self._params:
            setattr(self, p.name, p.interpret(params))

class CnnGenome1(object):
    """
    A genome for generalized neural networks.

    Terminology
        pin: Point at which the network is conceptually connected to the external world;
             pins are either input or output.
        node: Analog of a physical neuron.
        connection: Connection between a pin/node output and a node's input, or between a node's
             output and a pin/node input.
        key: Identifier for an object, unique within the set of similar objects.

    Design assumptions and conventions.
        1. Each output pin is connected only to the output of its own unique
           neuron by an implicit connection with weight one. This connection
           is permanently enabled.
        2. The output pin's key is always the same as the key for its
           associated neuron.
        3. Output neurons can be modified but not deleted.
        4. The input values are applied to the input pins unmodified.
    """

    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = DefaultNodeGene
        param_dict['connection_gene_type'] = DefaultConnectionGene
        return CnnGenome1Config(param_dict)

    @classmethod
    def write_config(cls, f, config):
        config.save(f)

    def __init__(self, key):
        # Unique identifier for a genome instance.
        self.key = key

        self.classification = DefaultGenome(0)

        # Fitness results.
        self.fitness = None

    def configure_new(self, config):
        """Configure a new genome based on the given configuration."""
        self.classification.configure_new(config)

    def configure_crossover(self, genome1, genome2, config):
        self.classification.configure_crossover(genome1.classification, genome2.classification, config)

    def mutate(self, config):
        """ Mutates this genome. """
        self.classification.mutate(config)

    def distance(self, other, config):
        """
        Returns the genetic distance between this genome and the other. This distance value
        is used to compute genome compatibility for speciation.
        """
        return self.classification.distance(
            other.classification, config)

    def size(self):
        """
        Returns genome 'complexity', taken to be
        (number of nodes, number of enabled connections)
        """
        return self.classification.size()

    def __str__(self):
        s = "Classification\n"
        s += self.classification.__str__()
        return s

    def set_fitness(self, fitness):
        self.fitness = fitness
        self.classification.fitness = fitness
