"""
2-input XOR example -- this is most likely the simplest possible example.
"""
from __future__ import print_function


import os
import neat
import numpy as np
import torch
import torch.nn as nn

from NeCNN import visualize

from NeCNN.Method1.genome import NECnnGenome_M1
from NeCNN.pytorch_converter import TorchCNN
from mnist import MNIST

# 2-input XOR inputs and expected outputs.

num_samples = 10
mndata = MNIST('../samples')
images, labels = mndata.load_training()
img = np.array(images, np.float32)
img = img.reshape(-1, 28, 28)
img = np.expand_dims(img, axis=1)
x_train = torch.from_numpy(img[:num_samples])
y_train = torch.from_numpy(np.array(labels[:num_samples]).astype(np.int64))

import torch.optim as optim

def classify(outputs):
    _, predicted = torch.max(outputs, 1)
    return predicted

def classification_error(outputs):
    classification_error = (classify(outputs) == y_train).sum() / len(y_train.numpy())
    return classification_error


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = TorchCNN.create_M1(genome, config)
        # visualize.draw_net(config.genome_config.classification_genome_config, genome.classification, True)
        loss = train_pytorch(net)
        output = net.forward(x_train)
        print(f"classification Error: {classification_error(output)}")
        #print(output)
        #print(y_train)
        genome.set_fitness(1/(1+loss.item()))

def train_pytorch(net, iterations = 1000):
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    loss = 100
    for i in range(iterations):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # print statistic
        if i % 100 == 0:  # print every 100 mini-batches
            print(f'[ {i + 1:5d}] loss: {loss.item():.3f}')

    return loss

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

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 20)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = TorchCNN.create_M1(winner, config)
    output = winner_net.forward(x_train)
    criterion = nn.CrossEntropyLoss()
    print(f"loss = {criterion(output, y_train)}")

    print('\nTraining\n')
    train_pytorch(winner_net)

    print('\nOutput:')
    output = winner_net.forward(x_train)
    criterion = nn.CrossEntropyLoss()
    print(f"loss = {criterion(output, y_train)}")

    visualize.draw_net(config.genome_config.classification_genome_config, winner.classification, True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward-cnn1')
    run(config_path)