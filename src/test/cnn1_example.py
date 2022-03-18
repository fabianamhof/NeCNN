"""
2-input XOR example -- this is most likely the simplest possible example.
"""
from __future__ import print_function


import os
import neat

from src.NeCNN import visualize
from src.NeCNN.cnn_genome_1 import *
from src.NeCNN.pytorch_neat import *


# 2-input XOR inputs and expected outputs.
xor_inputs = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
xor_outputs = np.array([[0.0],[1.0],[1.0],[0.0]])

import torch.optim as optim

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.set_fitness(4.0)
        net = TorchFeedForwardNetwork.create(genome.classification, config)
        output = net.forward(xor_inputs)
        for i, xo in enumerate(xor_outputs):
            genome.fitness -= ((output[i] - xo[0]) ** 2).item()
            genome.classification.fitness -= ((output[i] - xo[0]) ** 2).item()

def train_pytorch(net, iterations = 10000):
    if not torch.is_tensor(xor_outputs):
        labels = torch.from_numpy(xor_outputs).float()

    net.train()
    optimizer = optim.SGD(net.parameters(), lr=0.1)
    criterion = nn.BCELoss()
    prev_loss = 100
    for iter in range(iterations):
        net.zero_grad()
        outputs = net.forward(xor_inputs)
        loss = criterion(outputs, labels)
        if(prev_loss - loss < 0.001):
            break
        loss.backward()
        optimizer.step()

def run(config_file):
    # Load configuration.
    config = neat.Config(CnnGenome1, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = TorchFeedForwardNetwork.create(winner.classification, config)
    output = winner_net.forward(xor_inputs)
    for i, x in enumerate(zip(xor_inputs, xor_outputs)):
        print("input {!r}, expected output {!r}, got {!r}".format(x[0], x[1], output[i]))

    print('\nTraining\n')
    train_pytorch(winner_net)

    print('\nOutput:')
    output = winner_net.forward(xor_inputs)
    for i, x in enumerate(zip(xor_inputs, xor_outputs)):
        print("input {!r}, expected output {!r}, got {!r}".format(x[0], x[1], output[i]))

    node_names = {-1: 'A', -2: 'B', 0: 'A XOR B'}
    visualize.draw_net(config, winner.classification, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward-cnn1')
    run(config_path)