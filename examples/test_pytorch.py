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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = models.vgg16_bn(pretrained=True).to(device)
for param in net.features.parameters():
    param.requires_grad = False

num_ftrs = list(net.classifier.children())[0].in_features

classifier = [nn.Linear(in_features=num_ftrs, out_features=4096).to(device),
              nn.ReLU(inplace=True).to(device),
              nn.Dropout(p=0.5, inplace=False).to(device),
              nn.Linear(in_features=4096, out_features=2048).to(device),
              nn.ReLU(inplace=True).to(device),
              nn.Dropout(p=0.5, inplace=False).to(device),
              nn.Linear(in_features=2048, out_features=1024).to(device),
              nn.ReLU(inplace=True).to(device),
              nn.Dropout(p=0.5, inplace=False).to(device),
              nn.Linear(in_features=1024, out_features=10, bias=True).to(device)]

net.classifier = nn.Sequential(*classifier)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
epoch_loss = train_pytorch(net, optimizer, criterion, trainloader, device=device, epochs=3)
loss = get_loss(net, criterion, trainloader, device=device)
print(f"Epoch Loss {epoch_loss}; Loss: {loss};")
print(f"Classification: {classification_error(net, testloader, device=device)}")

torch.save(net, 'models/model.pth')
