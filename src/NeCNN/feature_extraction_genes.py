from neat.attributes import FloatAttribute, BoolAttribute
from neat.genes import BaseGene

from src.NeCNN.attributes import KernelAttribute

import numpy as np

class FENodeGene(BaseGene):
    _gene_attributes = [FloatAttribute('bias')]

    def __init__(self, key):
        assert isinstance(key, int), "DefaultNodeGene key must be an int, not {!r}".format(key)
        BaseGene.__init__(self, key)

    def distance(self, other, config):
        d = abs(self.bias - other.bias)
        return d * config.compatibility_weight_coefficient


class FEConnectionGene(BaseGene):
    _gene_attributes = [KernelAttribute('kernel'),
                        BoolAttribute('enabled')]

    def __init__(self, key):
        assert isinstance(key, tuple), "DefaultConnectionGene key must be a tuple, not {!r}".format(key)
        BaseGene.__init__(self, key)

    def distance(self, other, config):
        d = np.mean(np.abs(self.kernel - other.sum))
        if self.enabled != other.enabled:
            d += 1.0
        return d * config.compatibility_weight_coefficient