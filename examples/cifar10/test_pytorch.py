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

from NeCNN.Pytorch.resnet import ResNet18
from NeCNN.Pytorch.net import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainloader, testloader = load_data()

net = Net()
resnet = ResNet18(3)
resnet_features = [resnet.conv1, resnet.bn1, resnet.layer1, resnet.layer2, resnet.layer3,
                   resnet.layer4]
net.features = nn.Sequential(*resnet_features)
print(resnet)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet.parameters(), lr=learning_rate, momentum=momentum)
t1 = time.time()
epoch_loss = train_pytorch(resnet, optimizer, criterion, trainloader, device=device)
t2 = time.time()
print(t2 - t1)
loss = get_loss(resnet, criterion, trainloader, device=device)
print(f"Epoch Loss {epoch_loss}; Loss: {loss};")
print(f"Classification: {classification_error(resnet, testloader, device=device)}")

torch.save(net, 'models/model.pth')
