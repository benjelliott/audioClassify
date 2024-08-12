import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from train.py import train1
from plot.py import plot1

class Our_CNN(nn.Module):
    def __init__(self):
        # Initialize the nn.Module
        super().__init__()

        # Define our convolutions, pool and functions
        self.conv1 = nn.Conv2d(1, 2, 10)
        self.conv2 = nn.Conv2d(2, 6, 5)
        self.conv3 = nn.Conv2d(6, 10, 5)

        self.pool = nn.MaxPool2d(2, 2)

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout5 = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(660, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)


    def forward(self, x):
        # first convolutions and pooling

        x = self.dropout1(self.pool(F.relu(self.conv1(x))))

        x = self.dropout1(self.pool(F.relu(self.conv2(x))))

        x = self.dropout1(self.pool(F.relu(self.conv3(x)))
)
        # flatten

        x = torch.flatten(x, 1)

        # activation and fully-connected layers
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout1(F.relu(self.fc2(x)))
        x = self.dropout1(F.relu(self.fc3(x)))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)

        return x

our_net = Our_CNN()
print(our_net)