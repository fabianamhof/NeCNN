"""
Module that holds information about the training parameters and the dataset.
Also provides a function load_data() that returns the dataloader classes for both, the original data and the extracted features.
"""
import gzip
import os
import shutil
import tarfile

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url

from NeCNN.Pytorch.pytorch_helper import prepare_data
from small_norb_wrapper.dataset import SmallNORBDataset

num_workers = 8
train_batch_size = 128
test_batch_size = 128
learning_rate = 0.01
momentum = 0.9

transform = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

data_dir = './data/smallnorb_export'


def download_data(path, dest):
    if not os.path.exists(f"./data/{dest}"):
        dataset_url = path
        download_url(dataset_url, './data')
        # Extract from archive
        with gzip.open(f'./data/{dest}.gz', 'rb') as s_file, open(
                f"./data/{dest}",
                'wb') as d_file:
            shutil.copyfileobj(s_file, d_file)


files = {
    "smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat": "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz",
    "smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat": "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz",
    "smallnorb-5x46789x9x18x6x2x96x96-training-info.mat": "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz",
    "smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat": "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz",
    "smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat": "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz",
    "smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat": "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz",
}

for dest, path in files.items():
    download_data(path, dest)

if not os.path.exists(data_dir):
    # Initialize the dataset from the folder in which
    # dataset archives have been uncompressed
    dataset = SmallNORBDataset(dataset_root='./data')
    # Dump all images to disk
    dataset.export_to_jpg(export_dir=data_dir)

print(os.listdir(data_dir))
classes = os.listdir(data_dir + "/train")
print(classes)

# Look into the data directory

trainset = ImageFolder(data_dir + '/train', transform)
testset = ImageFolder(data_dir + '/test', transform)


def _get_path(path_to_model, type):
    s = path_to_model.split("/")
    s = s[-1].split(".")
    s = s[0]
    return f"./data/snorb_{s}_{type}_features"


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
