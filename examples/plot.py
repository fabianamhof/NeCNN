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
from NeCNN.Method1.genome import NECnnGenome_M1
from NeCNN.Pytorch.pytorch_converter import create_CNN
from NeCNN.Pytorch.pytorch_helper import classification_error, train_pytorch

folder = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device {device}")
mnist_mean = 0.1307
mnist_sd = 0.3081
num_workers = 8
train_batch_size = 128
classification_batch_size = 2048
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mnist_mean,), (mnist_sd,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=classification_batch_size, shuffle=False)


def run(config_file):
    # Load configuration.
    config = neat.Config(NECnnGenome_M1, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

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
    winner_net = create_CNN(winner, config.genome_config)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(winner_net.parameters(), lr=0.02, momentum=0.5)
    loss = train_pytorch(winner_net, optimizer, criterion, trainloader, device=device, printing_offset=-1)

    print(f'\nLoss: {loss}; Classification Error: {classification_error(winner_net, testloader, device=device)}')

    visualize.draw_net(config.genome_config.classification_genome_config, winner.classification,
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
    from data_info import *

    config_path = os.path.join(cwd, 'config')
    run(config_path)