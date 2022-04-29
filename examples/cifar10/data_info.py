"""
Module that holds information about the training parameters and the dataset.
Also provides a function load_data() that returns the dataloader classes for both, the original data and the extracted features.
"""
import os

import torch
import torchvision
import torchvision.transforms as transforms

from NeCNN.Pytorch.pytorch_helper import prepare_data

num_workers = 8
train_batch_size = 256
test_batch_size = 1028
learning_rate = 0.1
momentum = 0.9

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


def _get_path(path_to_model, type):
    s = path_to_model.split("/")
    s = s[-1].split(".")
    s = s[0]
    return f"./data/cifar10_{s}_{type}_features"


def load_data(path_to_model):
    """
    Loads the extracted features from the given model
    :param path_to_model: path to the model that is responsible for the feature extraction.
    :return: train and testloader for original data and extracted features.
    """
    if not (os.path.exists(_get_path(path_to_model, "train")) and os.path.exists(_get_path(path_to_model, "test"))):
        store_data(path_to_model)

    train_features = torch.load(_get_path(path_to_model, "train"))
    test_features = torch.load(_get_path(path_to_model, "test"))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True,
                                              num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False,
                                             num_workers=num_workers)

    trainloader_features = torch.utils.data.DataLoader(train_features, batch_size=train_batch_size, shuffle=True,
                                                       num_workers=num_workers)

    testloader_features = torch.utils.data.DataLoader(test_features, batch_size=test_batch_size, shuffle=True,
                                                      num_workers=num_workers)
    return trainloader, testloader, trainloader_features, testloader_features


def store_data(path_to_model):
    net = torch.load(path_to_model, map_location="cpu")

    print("Preparing data by already computing outputs of the feature extraction part...")

    train_features, test_features = prepare_data(
        net.features, trainset, testset,
        train_batch_size, test_batch_size,
        num_workers)

    torch.save(train_features, _get_path(path_to_model, "train"))
    torch.save(test_features, _get_path(path_to_model, "test"))

    print("...done and saved")
