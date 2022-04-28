"""
2-input XOR example -- this is most likely the simplest possible example.
"""
from __future__ import print_function

import os
import pickle
import sys

import neat
import shutil

import torch
from torch import nn, optim

from NeCNN.Method1.genome import NECnnGenome_M1
from NeCNN.Pytorch.pytorch_converter import create_CNN
from NeCNN.Pytorch.pytorch_helper import classification_error, train_pytorch

folder = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device {device}")


def eval_genome(genome, config):
    net = create_CNN(genome, config.genome_config)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    loss = train_pytorch(net, optimizer, criterion, trainloader, device=device, printing_offset=-1)
    fitness = 1 / (1 + loss)
    print(
        f"Genome: {genome.key} Loss: {loss} Fitness: {fitness}")
    genome.set_fitness(fitness)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        eval_genome(genome, config)


def run(config_file):
    # Load configuration.
    config = neat.Config(NECnnGenome_M1, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    checkpoints = neat.Checkpointer(generation_interval=10, time_interval_seconds=300,
                                    filename_prefix=f"{folder}/neat-checkpoint-")
    p.add_reporter(checkpoints)

    # Run for up to 100 generations.
    winner = p.run(eval_genomes, 50)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    winner_net = create_CNN(winner, config.genome_config)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(winner_net.parameters(), lr=0.02, momentum=0.5)
    loss = train_pytorch(winner_net, optimizer, criterion, trainloader, device=device, printing_offset=-1)

    print(f'\nClassification Error: {classification_error(winner_net, testloader, device=device)}')  #

    with open(f'{folder}/stats.pickle', 'wb') as f:
        pickle.dump(stats, f)
    with open(f'{folder}/winner.pickle', 'wb') as f:
        pickle.dump(winner, f)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    folder = sys.argv[1]
    isExist = os.path.exists(folder)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(folder)

    cwd = os.getcwd()
    sys.path.append(cwd)
    from data_info import *

    config_path = os.path.join(cwd, 'config')
    shutil.copyfile(config_path, f"{cwd}/{folder}/config")
    run(config_path)
