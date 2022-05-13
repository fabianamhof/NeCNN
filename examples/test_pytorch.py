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
trainloader, testloader, trainloader_features, testloader_features = load_data("model_good.pth")

# net = Net()
# net_cifar = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)
# net.features = net_cifar.features
# net.classifier = net_cifar.classifier
net = torch.load("models/model_good.pth", map_location="cpu")
#
# resnet = models.resnet18(pretrained=True).to(device)
# print(resnet)
# resnet_features = [resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4, resnet.avgpool]
# net.features = nn.Sequential(*resnet_features)
#
# for param in net.features.parameters():
#     param.requires_grad = False
#
#
# #num_ftrs = list(net.classifier.children())[0].in_features
# #print(num_ftrs)
#
# classifier = [nn.Linear(512, 10).to(device)]
#
# #input_lastLayer = net.classifier[-1].in_features
# #classifier = list(net.classifier.children())[:-1] # Remove last layer
# #classifier.extend([nn.Linear(input_lastLayer, 10).to(device)]) # Add our layer with 10 outputs
# net.classifier = nn.Sequential(*classifier)
print(net.classifier)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
t1 = time.time()
epoch_loss = train_pytorch(net, optimizer, criterion, trainloader, device=device)
t2 = time.time()
print(t2 - t1)
loss = get_loss(net, criterion, trainloader, device=device)
print(f"Epoch Loss {epoch_loss}; Loss: {loss};")
print(f"Classification: {classification_error(net, testloader, device=device)}")

torch.save(net, 'models/model.pth')
