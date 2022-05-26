import time

import torch


def predict(net, inputs, device='cpu'):
    """
    Forwards inputs through network and returns predicted labels
    """
    net.eval()
    if not torch.is_tensor(inputs):
        inputs = torch.from_numpy(inputs)
    net.to(device)
    inputs.to(device)
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    return predicted


def classification_error(net, dataloader, batches=-1, device='cpu'):
    """
    Calculates accuracy of the network
    :param net: Network
    :param dataloader: dataset
    :param batches: num_batches, -1 means all batches
    :return: accuracy
    """
    net.eval()
    net.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i == batches:
                return correct / total
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def get_loss(net, criterion, dataloader, device='cpu'):
    """
    Calculates the loss on the given dataset
    :param net: Network to evaluate
    :param criterion: Criterion
    :param dataloader: Dataset
    :return: Loss
    """
    net.eval()
    net.to(device)
    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.sampler)


def train_pytorch(net, optimizer, criterion, dataloader, printing_offset=500, device='cpu', epochs=2, scheduler=None):
    """
    Trains the network on the given dataset
    :param net: Network to train
    :param optimizer: Optimizer
    :param criterion: Criterion
    :param dataloader: Dataset
    :param printing_offset: Default 500, prints state every 500 batches
    :param epochs: num_epochs to train
    :return: epoch loss of the last epoch.
    """
    net.to(device)
    net.train()
    epoch_loss = 0.0
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            running_loss += loss.item()
            epoch_loss += loss.item() * inputs.size(0)

            loss.backward()
            optimizer.step()

            if i % printing_offset == 0 and printing_offset != -1:  # print every 2000 mini-batches
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(inputs), len(dataloader.dataset),
                           100. * i / len(dataloader), running_loss / min(i + 1, printing_offset)))
                running_loss = 0.0
        if scheduler is not None:
            scheduler.step()
    net.eval()

    return epoch_loss / len(dataloader.sampler)


def pretrain_features(net, dataloader, device='cpu'):
    """
    Forwards the given dataset on the given netowork
    :param net: Network
    :param dataloader: Dataset
    :return: Tuple of results of the network and labels
    """
    results = None
    labels = None
    for i, data in enumerate(dataloader):
        images, label = data[0], data[1]
        net.to(device)
        net.eval()
        with torch.no_grad():
            result = net(images.to(device))
            result = torch.flatten(result, 1).cpu()
            if results is None:
                labels = label
                results = result
            else:
                labels = torch.cat((labels, label))
                results = torch.cat((results, result))

    return results, labels


def prepare_data(net, trainset, testset, train_batch_size, test_batch_size, num_workers):
    """
    Uses the network to compute the outputs.
    :param net: Network
    :param trainset: Train Dataset
    :param testset: Test Dataset
    :param train_batch_size: Batch_size for training set evaluation
    :param test_batch_size: Batch size for test set evaluation
    :param num_workers: num_workers for the dataloaders
    :return: outputs of the network for trainset and testset
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True,
                                              num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False,
                                             num_workers=num_workers)

    trainset_features, trainset_labels = pretrain_features(net, trainloader, device=device)
    train_features = torch.utils.data.TensorDataset(trainset_features, trainset_labels)

    testset_features, testset_labels = pretrain_features(net, testloader, device=device)
    test_features = torch.utils.data.TensorDataset(testset_features, testset_labels)

    return train_features, test_features
