"""Handles genomes (individuals in the population)."""
from __future__ import division, print_function

import torch
from .classification_genome import *
from neat.config import ConfigParameter

from NeCNN.Pytorch.net import Net


def filter_params(keyword, params):
    filtered_params = dict()
    for key, value in params.items():
        if key.startswith(keyword):
            filtered_params[key[len(keyword):]] = value

    return filtered_params


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path):
    model = torch.load(path, map_location=device)
    return model


def lock_FE(model):
    # Prevent the trained weights from being modified
    for param in model.features.parameters():
        param.requires_grad = False
    return model


def get_num_features(model):
    return list(model.classifier.children())[0].in_features


class NECnnGenomeConfig_M1(object):
    allowed_connectivity = ['unconnected', 'fs_neat_nohidden', 'fs_neat', 'fs_neat_hidden',
                            'full_nodirect', 'full', 'full_direct',
                            'partial_nodirect', 'partial', 'partial_direct']

    def __init__(self, params):
        self._params = [ConfigParameter('feature_extraction_model_path', str),
                        ConfigParameter('image_channels', int),
                        ConfigParameter('image_width', int),
                        ConfigParameter('image_height', int)]
        for p in self._params:
            setattr(self, p.name, p.interpret(params))
        self.feature_extraction_model = lock_FE(load_model(self.feature_extraction_model_path))
        classification_params = filter_params("classification_", params)
        classification_params["num_inputs"] = get_num_features(self.feature_extraction_model)
        self.classification_genome_config = ClassificationGenome.parse_config(classification_params)


class NECnnGenome_M1(object):

    @classmethod
    def parse_config(cls, param_dict):
        return NECnnGenomeConfig_M1(param_dict)

    def __init__(self, key):
        # Unique identifier for a genome instance.
        self.key = key

        self.classification = ClassificationGenome(0)

        # Fitness results.
        self.fitness = None

    def configure_new(self, config):
        """Configure a new genome based on the given configuration."""
        self.classification.configure_new(config.classification_genome_config)

    def configure_crossover(self, genome1, genome2, config):
        self.classification.configure_crossover(genome1.classification, genome2.classification,
                                                config.classification_genome_config)

    def mutate(self, config):
        """ Mutates this genome. """
        self.classification.mutate(config.classification_genome_config)

    def distance(self, other, config):
        """
        Returns the genetic distance between this genome and the other. This distance value
        is used to compute genome compatibility for speciation.
        """
        return self.classification.distance(
            other.classification, config.classification_genome_config)

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
