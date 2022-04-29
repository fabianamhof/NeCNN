"""
2-input XOR example -- this is most likely the simplest possible example.
"""
from __future__ import print_function

import os
import pickle
import sys

import neat
import shutil

import multiprocessing as mp

import torch
from torch import nn
from torch import optim
import torchvision
import torchvision.transforms as transforms

import numpy as np

from NeCNN import visualize
from NeCNN.Method1.genome import ClassificationGenome
from NeCNN.Pytorch.pytorch_converter import TorchFeedForwardNetwork, create_CNN
from NeCNN.Pytorch.pytorch_helper import classification_error, train_pytorch

folder = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device {device}")

trainloader, testloader, trainloader_features, testloader_features = None, None, None, None


def run(config_file):
    # Load configuration.
    config = neat.Config(ClassificationGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    global trainloader, testloader, trainloader_features, testloader_features
    trainloader, testloader, trainloader_features, testloader_features = load_data(
        config.genome_config.feature_extraction_model)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    with open(f"{folder}/stats.pickle", 'rb') as file:
        stats = pickle.load(file)

    with open(f"{folder}/winner.pickle", 'rb') as file:
        winner = pickle.load(file)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    winner_net = TorchFeedForwardNetwork.create(winner, config.genome_config)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(winner_net.parameters(), lr=learning_rate, momentum=momentum)
    loss = train_pytorch(winner_net, optimizer, criterion, trainloader_features, device=device, printing_offset=-1)

    pretrained = torch.load(config.genome_config.feature_extraction_model, map_location=device)
    winner_cnn = create_CNN(features=pretrained.features, classifier=winner_net)
    print(f'\nLoss: {loss}; Classification Error: {classification_error(winner_cnn, testloader, device=device)}')

    visualize.draw_net(config.genome_config, winner,
                       filename=f"{folder}/net", view=False)
    visualize.plot_stats(stats, ylog=False, filename=f"{folder}/avg_fitness.svg", view=False)
    visualize.plot_species(stats, filename=f"{folder}/speciation.svg", view=False)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    folder = sys.argv[1]

    cwd = os.getcwd()
    sys.path.append(cwd)

    config_path = os.path.join(cwd, f'{folder}/config')
    from data_info import *

    run(config_path)
