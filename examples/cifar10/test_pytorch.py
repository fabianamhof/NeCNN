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

from NeCNN.Pytorch.resnet import resnet50
from NeCNN.Pytorch.net import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainloader, testloader = load_data()

net = Net()
resnet = resnet50(pretrained=False, num_classes=10)
resnet.load_state_dict(torch.load("models/resnet50.pt"))
resnet_features = [resnet.conv1, resnet.bn1,resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3,
                   resnet.layer4, resnet.avgpool]
net.features = nn.Sequential(*resnet_features)
net.classifier = resnet.fc
#print(net)
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
#t1 = time.time()
#epoch_loss = train_pytorch(net, optimizer, criterion, trainloader, device=device, epochs=15)
#t2 = time.time()
#print(t2 - t1)
loss = get_loss(net, criterion, trainloader, device=device)
print(f"Epoch Loss ; Loss: {loss};")
print(f"Classification: {classification_error(net, testloader, device=device)}")

torch.save(net, 'models/resnet50.pth')