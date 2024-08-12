import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from train.py import train1
from plot.py import plot1

# getting dataset of audio files

batch_size = 4

def make_dataset(train = bool):
    traintest = "train" if train else "test"
    load = lambda name: torch.from_numpy(np.load(f"./{name}{traintest}_mel.npy"))
    X = torch.add(torch.multiply(load("X").unsqueeze(dim = 1), 2), -1)
    y = load("y")

    dataset = torch.utils.data.TensorDataset(X,y)
    return torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = train, num_workers = 2)

trainloader = make_dataset(train = True)

testloader = make_dataset(train = False)

classes = ("air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music")

# defining CNN

class Our_CNN(nn.Module):
    def __init__(self):
        # Initialize the nn.Module
        super().__init__()

        # Define our convolutions, pool and functions
        self.conv1 = nn.Conv2d(1, 32, 10)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.dropout25 = nn.Dropout(p=0.25)
        self.dropout5 = nn.Dropout(p=0.5)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(134976, 128)
        self.fc2 = nn.Linear(128, 10)



    def forward(self, x):
        # first convolutions and pooling
        x = F.relu(self.conv1(x))
        x = self.dropout25(self.pool(F.relu(self.conv2(x))))


        # flatten
        x = torch.flatten(x, 1)

        # activation and fully-connected layers
        x = self.dropout5(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))

        return x

our_net = Our_CNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(our_net.parameters(), lr = 0.001)

train1(our_net, trainloader, criterion, optimizer)