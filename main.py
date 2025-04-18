import torch
import torchvision.transforms as transforms
import torchvision
import torch.optim as optim
import torch.nn as nn

from model import Net
from train import train
from test import test

if __name__ == '__main__':
    transform = transforms.ToTensor()
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000,
                                              shuffle=False)
    model = Net().cuda()
    criterion = nn.CrossEntropyLoss()   
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train(model, trainloader, optimizer, criterion, epochs=epochs, device=device)

    test(model, testloader, device=device)