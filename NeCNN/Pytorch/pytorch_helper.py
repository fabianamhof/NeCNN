import time

import torch


def predict(net, inputs, device='cpu'):
    net.eval()
    if not torch.is_tensor(inputs):
        inputs = torch.from_numpy(inputs)
    net.to(device)
    inputs.to(device)
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    return predicted


def classification_error(net, dataloader, batches=-1, device='cpu'):
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


def train_pytorch(net, optimizer, criterion, dataloader, printing_offset=500, device='cpu', epochs=2):
    net.to(device)
    net.train()
    start = time.perf_counter()
    epoch_loss = 0.0
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # start2 = time.perf_counter()
            outputs = net(inputs)
            # print(f'Forward {time.perf_counter() - start2}s')
            # start2 = time.perf_counter()
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            epoch_loss += loss.item() * inputs.size(0)
            # print(f'Loss {time.perf_counter() - start2}s')
            # start2 = time.perf_counter()
            loss.backward()
            optimizer.step()
            # print(f'Backward {time.perf_counter() - start2}s')

            if i % printing_offset == 0 and printing_offset != -1:  # print every 2000 mini-batches
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(inputs), len(dataloader.dataset),
                           100. * i / len(dataloader), running_loss / min(i + 1, printing_offset)))
                running_loss = 0.0
    net.eval()
    # print(f'Finished Training in {time.perf_counter() - start}s')
    return epoch_loss / len(dataloader.sampler)


def pretrain_features(net, dataloader, device='cpu'):
    results = None
    labels = None
    for i, data in enumerate(dataloader):
        images, label = data[0], data[1]
        net.to(device)
        net.eval()
        with torch.no_grad():
            result = net(images.to(device))
            result = torch.flatten(result, 1)
            if results is None:
                labels = label
                results = result
            else:
                labels = torch.cat((labels, label))
                results = torch.cat((results, result))

    return results, labels


def prepare_data(net, trainset, testset, train_batch_size, test_batch_size, num_workers):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True,
                                              num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False)

    trainset_features, trainset_labels = pretrain_features(net, trainloader, device=device)
    train_features = torch.utils.data.TensorDataset(trainset_features, trainset_labels)

    testset_features, testset_labels = pretrain_features(net, testloader, device=device)
    test_features = torch.utils.data.TensorDataset(testset_features, testset_labels)

    return train_features, test_features
