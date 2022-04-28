import torch
import torchvision
import torchvision.transforms as transforms

num_workers = 8

train_batch_size = 128
classification_batch_size = 2048

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Create the data loaders for the train and test sets
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
trainloader_2 = torch.utils.data.DataLoader(trainset, batch_size=classification_batch_size, shuffle=True,
                                            num_workers=num_workers)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=classification_batch_size, shuffle=False)

learning_rate = 0.02
momentum = 0.5
