"""Handles node and connection genes."""
import warnings
from random import random
from neat.attributes import BoolAttribute
from neat.genes import BaseGene


class NeCnnNodeGene(BaseGene):
    _gene_attributes = []

    def __init__(self, key):
        assert isinstance(key, int), "DefaultNodeGene key must be an int, not {!r}".format(key)
        BaseGene.__init__(self, key)

    def distance(self, other, config):
        return 0


class NeCnnConnectionGene(BaseGene):
    _gene_attributes = [BoolAttribute('enabled')]

    def __init__(self, key):
        assert isinstance(key, tuple), "DefaultConnectionGene key must be a tuple, not {!r}".format(key)
        BaseGene.__init__(self, key)

    def distance(self, other, config):
        d = 0.0
        if self.enabled != other.enabled:
            d = 1.0
        return d
