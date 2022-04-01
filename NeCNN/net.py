import torch
from torch import nn
import time

features = [
    nn.Conv2d(1, 20, 5),
    nn.MaxPool2d(2, 2),
    nn.ReLU(),
    nn.Conv2d(20, 20, 5),
    nn.MaxPool2d(2, 2),
    nn.ReLU(),
]

classifier = [
    nn.Linear(320, 50),
    nn.ReLU(),
    nn.Linear(50, 10),
    nn.ReLU()
]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        #start = time.perf_counter()
        x = self.features(x)
        #end = time.perf_counter()
        #print(f"Feature extraction: {end - start}")
        #start = time.perf_counter()
        x = torch.flatten(x, 1)
        #end = time.perf_counter()
        #print(f"Flattening: {end - start}")
        #start = time.perf_counter()
        x = self.classifier(x)
        #end = time.perf_counter()
        #print(f"Classification: {end - start}")
        return x