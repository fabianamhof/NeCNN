import torch
from torch import nn

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
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

net = Net()
net.load_state_dict(torch.load("./results/model.pth"))
torch.save(net, './results/model.pt') # Save