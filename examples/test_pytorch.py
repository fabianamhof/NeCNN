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

from NeCNN.Pytorch.net import Net

net = Net()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#net_cifar = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)

#net.features = net_cifar.features
#net.classifier = net_cifar.classifier
net = torch.load("models/model_good.pth")
for param in net.features.parameters():
    param.requires_grad = False

num_ftrs = list(net.classifier.children())[0].in_features
print(num_ftrs)
classifier = [nn.Linear(in_features=num_ftrs, out_features=512).to(device),
              nn.ReLU(inplace=True).to(device),
              nn.Dropout(p=0.5, inplace=False).to(device),
              nn.Linear(in_features=512, out_features=10).to(device)]

net.classifier = nn.Sequential(*classifier)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.classifier.parameters(), lr=learning_rate, momentum=momentum)
epoch_loss = train_pytorch(net.classifier, optimizer, criterion, trainloader, device=device)
loss = get_loss(net.classifier, criterion, trainloader, device=device)
print(f"Epoch Loss {epoch_loss}; Loss: {loss};")
print(f"Classification: {classification_error(net, testloader, device=device)}")

torch.save(net, 'models/model.pth')
