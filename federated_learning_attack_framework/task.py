"""federated-learning-attack-framework: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')

    Input layer takes a tensor with size (-1, 3, 28, 28)

    Output layer gives a tensor with size (-1, 10)

    
        ------------------------------------------------
            Layer (type)               Output Shape     
        ================================================
            Conv2d-1                 [-1, 6, 28, 28]    
            MaxPool2d-2              [-1, 6, 14, 14]    
            Conv2d-3                [-1, 16, 10, 10]    
            MaxPool2d-4               [-1, 16, 5, 5]    
            Linear-5                       [-1, 120]    
            Linear-6                        [-1, 84]    
            Linear-7                        [-1, 10]    
        ================================================

    """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Makes model forward pass from given Tensor x
        
        Examples:
            >>> net = Net()
            >>> net(torch.rand((1, 3, 32, 32)))
            tensor([[ 0.1002,  0.0072,  0.0449, -0.0056,  0.0240,  0.0754, -0.0188,  0.0601, 0.0735,  0.0297]], grad_fn=<AddmmBackward0>)

            >>> net = Net()
            >>> net.forward(torch.rand((1, 3, 32, 32)))
            tensor([[ 0.0778, -0.0811, -0.1095,  0.0640, -0.0621,  0.0139, -0.0363,  0.0157, 0.0502,  0.0098]], grad_fn=<AddmmBackward0>)


        Args:
            x: A tensor representing a -1x3x32x32 image

        Returns:
            A tensor representing softmax activation output of the model given x.
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)


fds = None  # Cache FederatedDataset


def load_data(partition_id: int, num_partitions: int):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        print('LOUTRE')
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


def train(net: nn.Module, trainloader: DataLoader, epochs: int, device: torch.device) -> float:
    """Train the model on the training set.
    
    Args:
        net: The model to train
        trainloader: The dataloader to train the model on
        epochs : The number of training epochs
        device : The device to use

    Returns:
        The average loss during the training.
    """
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Test the model on the test set with a single forward pass.
    
    Args:
        net: The model to test
        testloader: The dataloader to test the model on
        device : The device to use 

    Returns:
        The loss and the accuracy on the test dataloader.
    """
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(net: nn.Module)->list:
    """Get weights from model.
    
    Args:
        net: The model whiwh we want the wieghts

    Returns:
        The weights in a list of layers param
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    "Set weights"
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
