import os
import sys
import time

sys.path.append(os.getcwd())
from data_info import *

import torch
from torch import nn
from torch import optim
from torchvision import models

from NeCNN.Pytorch.pytorch_helper import classification_error, train_pytorch, get_loss

from NeCNN.Pytorch.resnet import resnet18
from NeCNN.Pytorch.net import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainloader, testloader = load_data()

net = Net()
resnet = resnet18(pretrained=False, num_classes=5)
conv1 = nn.Conv2d(1, 64, 3, stride=(1, 1), padding=(1, 1))
resnet_features = [conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3,
                   resnet.layer4, resnet.avgpool]
net.features = nn.Sequential(*resnet_features)
net.classifier = resnet.fc
print(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
t1 = time.time()
epoch_loss = train_pytorch(net, optimizer, criterion, trainloader, device=device, epochs=10)
t2 = time.time()
print(t2 - t1)
loss = get_loss(net, criterion, trainloader, device=device)
print(f"Epoch Loss {epoch_loss}; Loss: {loss};")
print(f"Classification: {classification_error(net, testloader, device=device)}")

torch.save(net, 'models/resnet18.pth')
