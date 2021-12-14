import json
import os
import random

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from NN import Net
from helpers import *


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
batch_size = 64


def get_transform():
    return transforms.Compose(
        [
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         ])

def dataset_loader(transform):

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform) #torchvision.datasets.ImageFolder(root='./dataset/train', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=1)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform) #torchvision.datasets.ImageFolder(root='./dataset/test', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=1)

    return trainloader, testloader


def train_net(train_dataset, epoch_number):
    for epoch in range(epoch_number):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_dataset, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


def test_net(test_dataset):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dataset:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def save(path):
    torch.save(net.state_dict(), path)


def load(path):
    net.load_state_dict(torch.load(path))


# Main function
if __name__ == '__main__':
    with open('data/test.json', 'r') as f:
        jej = json.load(f)
        print(jej)

    # image_preprocessing('data/test/fracture_real/' + random.choice(os.listdir('data/test/fracture_real')))

    # train, test = dataset_loader(get_transform())
    #
    # train_net(train, 2)
    # accuracy = test_net(test)
    # print("Network accuracy: ", accuracy)


