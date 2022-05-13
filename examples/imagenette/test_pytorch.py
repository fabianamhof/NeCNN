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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainloader, testloader, trainloader_features, testloader_features = load_data("models/model_good.pth")

net = models.resnet18(pretrained=True).to(device)

# net = Net()
# net_cifar = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)
# net.features = net_cifar.features
# net.classifier = net_cifar.classifier
# net = torch.load("models/model_good.pth", map_location=device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
loss = get_loss(net, criterion, trainloader, device=device)
print(f"Loss: {loss};")
print(f"Classification: {classification_error(net, testloader, device=device)}")

torch.save(net, 'models/model.pth')
