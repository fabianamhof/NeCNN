"""
Module that holds information about the training parameters and the dataset.
Also provides a function load_data() that returns the dataloader classes for both, the original data and the extracted features.
"""
import os

import torch
import torchvision
import tarfile
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url

from NeCNN.Pytorch.pytorch_helper import prepare_data

num_workers = 8
train_batch_size = 64
test_batch_size = 64
learning_rate = 0.01
momentum = 0.9

stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
transform = transforms.Compose(
    [
        transforms.Resize((254, 254)),
        transforms.ToTensor(),
        transforms.Normalize(*stats, inplace=True)
    ])

if not os.path.exists("./data/imagenette2-320"):
    dataset_url = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz'
    db = download_url(dataset_url, '.')
    # Extract from archive
    with tarfile.open('./imagenette2-320.tgz', 'r:gz') as tar:
        tar.extractall(path='./data')

# Look into the data directory
data_dir = './data/imagenette2-320'
print(os.listdir(data_dir))
classes = os.listdir(data_dir + "/train")
print(classes)

trainset = ImageFolder(data_dir + '/train', transform)
testset = ImageFolder(data_dir + '/val', transform)


def _get_path(path_to_model, type):
    s = path_to_model.split("/")
    s = s[-1].split(".")
    s = s[0]
    return f"./data/imagenette2_{s}_{type}_features"


def load_data(path_to_model=None):
    """
    Loads the extracted features from the given model
    :param path_to_model: path to the model that is responsible for the feature extraction.
    :return: train and testloader for original data and extracted features.
    """
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True,
                                              num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False,
                                             num_workers=num_workers)
    if (path_to_model is None):
        return trainloader, testloader

    if not (os.path.exists(_get_path(path_to_model, "train")) and os.path.exists(_get_path(path_to_model, "test"))):
        store_data(path_to_model)

    train_features = torch.load(_get_path(path_to_model, "train"))
    test_features = torch.load(_get_path(path_to_model, "test"))

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
