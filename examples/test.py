"""
2-input XOR example -- this is most likely the simplest possible example.
"""
from __future__ import print_function

import billiard as multiprocessing
import os
import neat

import torch
from torch import nn
from torch import optim
import torchvision
import torchvision.transforms as transforms

from NeCNN import visualize
from NeCNN.Method1.genome import NECnnGenome_M1, create_CNN


mnist_mean = 0.1307
mnist_sd = 0.3081
train_batch_size = 64
classification_batch_size = 1000
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mnist_mean,), (mnist_sd,))])

# Create the data loaders for the train and test sets
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)
trainloader_all = torch.utils.data.DataLoader(trainset, batch_size=classification_batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=classification_batch_size, shuffle=False, num_workers=2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def predict(net, inputs):
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    return predicted

def classification_error(net, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def eval_genome(genome, config):
    net = create_CNN(genome, config.genome_config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
    visualize.draw_net(config.genome_config.classification_genome_config, genome.classification, True)
    train_pytorch(net, optimizer, criterion, device)
    net.eval()
    images, labels = next(iter(trainloader_all))
    outputs = net(images)
    loss = criterion(outputs, labels)
    print(f"Classification Error: {classification_error(net, trainloader)}, Loss: {loss.item()}")
    genome.set_fitness(1 / (1 + loss.item()))

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        eval_genome(genome, config)

def train_pytorch(net, optimizer, criterion, device):
    net.train()
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 1999 == 0:  # print every 2000 mini-batches
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(inputs), len(trainloader.dataset),
                           100. * i / len(trainloader), loss.item()))

    print('Finished Training')

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

    pe = neat.ParallelEvaluator(3, eval_genome)
    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    winner_net = create_CNN(winner, config.genome_config)
    print(f'\nClassification Error: {classification_error(winner_net, testloader)}')

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