import time

import torch
from torch import nn
from torch import optim
import torchvision
import torchvision.transforms as transforms

from NeCNN.net import Net

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# Create the data loaders for the train and test sets
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=2)


def train_pytorch(net, optimizer, criterion, device):
    net.train()
    start = time.perf_counter()
    for epoch in range(2):  # loop over the dataset multiple times

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

            if i % 500 == 0:  # print every 2000 mini-batches
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(inputs), len(trainloader.dataset),
                           100. * i / len(trainloader), loss.item()))

    print(f'Finished Training in {time.perf_counter() - start}s')

def classification_error(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def load_model(path, python_class):
    model = python_class()
    model.load_state_dict(torch.load(path))
    return model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

net = load_model("./results/model_good.pth", Net)
net.to(device)
for param in net.features.parameters():
    param.requires_grad = False

classifier = [
    nn.Linear(320, 30),
    nn.ReLU(),
    nn.Linear(30, 10),
    nn.ReLU()
]

net.classifier = nn.Sequential(*classifier)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
train_pytorch(net, optimizer, criterion, device)
torch.save(net.state_dict(), 'results/model.pth')
print(f"Classification: {classification_error(net)}")
