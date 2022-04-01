"""
2-input XOR example -- this is most likely the simplest possible example.
"""
from __future__ import print_function

import copy
import time

import billiard as multiprocessing
import os
import neat

import torch
from torch import nn
from torch import optim
import torchvision
import torchvision.transforms as transforms

from NeCNN import visualize
from NeCNN.Method1.genome import NECnnGenome_M1
import NeCNN.pytorch_converter as converter1
import NeCNN.pytorch_converter_2 as converter2

def create_CNN(genome, config, converter):
    model = config.feature_extraction_model
    model_copy = copy.deepcopy(model)
    classifier = converter.TorchFeedForwardNetwork.create(genome.classification, config.classification_genome_config)
    model_copy.classifier = classifier
    return model_copy

mnist_mean = 0.1307
mnist_sd = 0.3081
train_batch_size = 10
classification_batch_size = 10
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mnist_mean,), (mnist_sd,))])

# Create the data loaders for the train and test sets
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)
trainloader_all = torch.utils.data.DataLoader(trainset, batch_size=classification_batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=classification_batch_size, shuffle=False, num_workers=2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def run(config_file):
    # Load configuration.
    config = neat.Config(NECnnGenome_M1, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    genome = NECnnGenome_M1(0)
    genome.configure_new(config.genome_config)
    criterion = nn.CrossEntropyLoss()
    visualize.draw_net(config.genome_config.classification_genome_config, genome.classification, True)
    cnn2 = create_CNN(genome, config.genome_config, converter2)
    optimizer2 = optim.SGD(cnn2.parameters(), lr=1, momentum=0.5)
    cnn1 = create_CNN(genome, config.genome_config, converter1)
    optimizer1 = optim.SGD(cnn1.parameters(), lr=1, momentum=0.5)
    images, labels = next(iter(trainloader_all))

    outputs2 = cnn2(images)
    loss = criterion(outputs2, labels)
    loss.backward()
    optimizer2.step()
    print(outputs2)
    outputs2 = cnn2(images)
    print(outputs2)

    outputs1 = cnn1(images)
    loss = criterion(outputs1, labels)
    loss.backward()
    optimizer1.step()
    print(outputs1)
    outputs1 = cnn1(images)
    print(outputs1)



if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward-cnn1')
    run(config_path)