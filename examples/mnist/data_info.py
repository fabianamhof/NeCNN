from xml.dom import NoModificationAllowedErr
import torch
import torchvision
import torchvision.transforms as transforms
import os

import neat


import torch
import torchvision
import torchvision.transforms as transforms

from NeCNN.Method1.genome import NECnnGenome_M1

num_workers = 0

train_batch_size = 128
classification_batch_size = 1028

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pretrain_feature_extraction(dataloader):
    results = None
    labels = None
    for i, data in enumerate(dataloader):
        images, label = data[0], data[1]
        cwd = os.getcwd()
        config_path = os.path.join(cwd, 'config')
        config = neat.Config(NECnnGenome_M1, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_path)
        
        model = config.genome_config.feature_extraction_model
        model.to(device)
        model.eval()
        result = model.features(images.to(device))
        result = torch.flatten(result, 1)
        if results is None:
            labels = label
            results = result
        else:
            labels = torch.cat((labels, label))
            results = torch.cat((results, result))
        
    return results, labels


mnist_mean = 0.1307
mnist_sd = 0.3081

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mnist_mean,), (mnist_sd,))])

# Create the data loaders for the train and test sets
mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_loader = torch.utils.data.DataLoader(mnist, batch_size=1024, shuffle=True, num_workers=num_workers)

print("Preparing data by already computing outputs of the feature extraction part...")
results, labels = pretrain_feature_extraction(mnist_loader)
print("...done")
testset = torch.utils.data.TensorDataset(results, labels)
trainloader = torch.utils.data.DataLoader(testset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
trainloader_2 = torch.utils.data.DataLoader(testset, batch_size=classification_batch_size, shuffle=True,
                                            num_workers=num_workers)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=classification_batch_size, shuffle=False)

learning_rate = 0.1
momentum = 0.9
