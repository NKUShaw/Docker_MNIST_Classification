import torch
import torchvision.transforms as transforms
import torchvision
import torch.optim as optim
import torch.nn as nn
import mlflow

from model import Net
from train import train
from test import test

if __name__ == '__main__':
    mlflow.set_experiment("MNIST-Classification")
    with mlflow.start_run():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        transform = transforms.ToTensor()
        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False,
                                            download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                                shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1000,
                                                shuffle=False)

        model = Net().to(device)
        criterion = nn.CrossEntropyLoss()   
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        epochs = 10
        

        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("batch_size", 64)
        mlflow.log_param("optimizer", "Adam")

        train(model, trainloader, optimizer, criterion, epochs=epochs, device=device)

        test_acc = test(model, testloader, device=device)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.pytorch.log_model(model, "model")
