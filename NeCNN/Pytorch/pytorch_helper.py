import time

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(net, inputs, device='cpu'):
    if not torch.is_tensor(inputs):
        inputs = torch.from_numpy(inputs)
    net.to(device)
    inputs.to(device)
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    return predicted


def classification_error(net, dataloader, batches=-1, device='cpu'):
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
