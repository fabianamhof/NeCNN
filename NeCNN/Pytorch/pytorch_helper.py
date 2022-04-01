import time

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(net, inputs):
    net.to(device)
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    return predicted

def classification_error(net, dataloader, batches = -1):
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

def train_pytorch(net, optimizer, criterion, dataloader, printing_offset = 500):
    net.to(device)
    net.train()
    start = time.perf_counter()
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            if i % printing_offset == 0:  # print every 2000 mini-batches
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(inputs), len(dataloader.dataset),
                           100. * i / len(dataloader), running_loss / min(i+1, printing_offset)))
                running_loss = 0.0
    net.eval()
    print(f'Finished Training in {time.perf_counter() - start}s')