#federated-learning-attack-framework/utils/task.py

"""Provide model class and all manipulation function.

This module allows the user to get the model to train and perform actions on it.

The module contains the following functions:

- `Net()` - The model's class.
- `train(net, trainloader, epochs, device)` - Train the model.
- `test(net, testloader, device)` - Test the model.
- `get_weights(net)` - Get the model state dict in a list.
- `set_weights(net, parameters)` - Set the model state dict from a given list.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')

    Input layer takes a tensor with size (-1, 3, 28, 28)

    Output layer gives a tensor with size (-1, 10)

    
        ------------------------------------------------
            Layer (type)               Output Shape     
        ================================================
            Conv2d                   [-1, 6, 28, 28]    
            MaxPool2d                [-1, 6, 14, 14]    
            Conv2d                  [-1, 16, 10, 10]    
            MaxPool2d                 [-1, 16, 5, 5]    
            Linear                         [-1, 120]    
            Linear                          [-1, 84]    
            Linear                          [-1, 10]    
        ================================================

    """

    def __init__(self) -> nn.Module:
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
            A tensor representing the softmax activation of the model's output layer given x.
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)


def train(net: nn.Module, trainloader: DataLoader, epochs: int, device: torch.device) -> float:
    """Train the model on the training set.

    Training loop :

        1. Get images and labels from batch
        2. Compute model output on image batch
        3. Compute output loss with label
        4. Make back propagation and optimizer step

    Fixed parameters :

        - Optimizer: Adam, lr=0.01
        - Loss : CrossEntropyLoss
    
    Args:
        net: The model to train
        trainloader: The dataloader to train the model on
        epochs: The number of training epochs
        device: The device to use (cpu or cuda)

    Returns:
        The average loss during the training.
    """
    # Move model to device
    net.to(device)

    # Setup loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    # Switch model to training mode
    net.train()

    running_loss = 0.0

    # Main loop
    for _ in range(epochs):
        # run over batches
        for batch in trainloader:
            # Extract images and label
            images = batch["img"]
            labels = batch["label"]
            # Reset optimizer gradient
            optimizer.zero_grad()
            # Compute loss and back propagate
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            # Make an optimization step
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net: nn.Module, testloader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Test the model on the test set with a single forward pass.
    
    Args:
        net: The model to test
        testloader: The dataloader to test the model on
        device: The device to use 

    Returns:
        The loss and the accuracy on the test dataloader.
    """
    # Move model to device
    net.to(device)
    
    # Setup loss functiona
    criterion = torch.nn.CrossEntropyLoss()

    correct, loss = 0, 0.0

    # No gradient computation (testing, should not affect model weights)
    with torch.no_grad():
        # Run over batches
        for batch in testloader:
            # Extract images and label
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            # Compute model output and loss
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            # Compute batch score
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

    # Copute global accracy and loss 
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)

    return loss, accuracy


def get_weights(net: nn.Module) -> list[float]:
    """Get weights from model.
    
    Args:
        net: The model which we want to get the weights

    Returns:
        The weights in a list of layers param
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net: nn.Module, parameters: list[float]) -> None:
    """Set mode weights from param list.
    
    Args:
        net: The model which we want to set the weights
        parameters: The new state dict in a list
    """
    # Bind keys and values
    params_dict = zip(net.state_dict().keys(), parameters)
    # transform in a dict
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    # Replace weigths
    net.load_state_dict(state_dict, strict=True)
