from neat.genome import *

from genes import FENodeGene, FEConnectionGene

class CNNGenomeConfig(DefaultGenomeConfig):
    def __init__(self, params):
        super().__init__(params)
        self._params += ConfigParameter('image_width', int)
        self._params += ConfigParameter('image_height', int)
        self._params += ConfigParameter('classification_outputs', int)


class CNNGenome(DefaultGenome):
    @classmethod
    def parse_config(cls, param_dict):
        param_dict['classification_node_gene_type'] = FENodeGene
        param_dict['classification_connection_gene_type'] = FEConnectionGene
        return DefaultGenomeConfig(param_dict)