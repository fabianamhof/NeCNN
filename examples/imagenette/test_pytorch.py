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
trainloader, testloader  = load_data()

net = Net()
resnet = models.resnet18(pretrained=True).to(device)
resnet_features = [resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4, resnet.avgpool]
net.features = nn.Sequential(*resnet_features)

for param in net.features.parameters():
    param.requires_grad = False

classifier = [nn.Linear(in_features=512, out_features=512).to(device), 
              nn.ReLU(inplace=True).to(device), 
              nn.Dropout(p=0.5, inplace=False).to(device), 
              nn.Linear(in_features=512, out_features=10).to(device)]

net.classifier = nn.Sequential(*classifier)
#net = Net()
# net_cifar = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)
# net.features = net_cifar.features
# net.classifier = net_cifar.classifier
# net = torch.load("models/model_good.pth", map_location=device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
epoch_loss = train_pytorch(net, optimizer, criterion, trainloader, device=device)
loss = get_loss(net, criterion, trainloader, device=device)
print(f"Epoch Loss: {epoch_loss} Loss: {loss};")
print(f"Classification: {classification_error(net, testloader, device=device)}")

torch.save(net, 'models/model.pth')

