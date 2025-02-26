#federated-learning-attack-framework/client.py

"""Provide client app.

This module allows the user to instantiate clients and launch them in the federation.

The module contains the following functions:

- `FlowerClient()` - The client's class.
- `client_fn(context)` - The function to instantiate clients.
"""

import torch
from torch import nn as nn
from torch.utils.data import DataLoader

from flwr.common import Context
from flwr.common.typing import Scalar
from flwr.client import ClientApp, NumPyClient

from federated_learning_attack_framework.utils.task import Net, get_weights, set_weights, test, train
from federated_learning_attack_framework.utils.data import load_data


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    """Simple client from NumpyClient class

    For each round where the client is triggered, it will follow tese steps:

    1. Replace its model with the recieived weights
    2. Train the model on its data
    3. Return back the new local weights, the training data loader length and the training loss
    """

    def __init__(self, net: nn.Module, trainloader: DataLoader, valloader: DataLoader, local_epochs: int):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters: list[float], config: dict[str, Scalar]) -> tuple[list[float], int, dict[str, Scalar]]:
        """ Train the recieved model on local data.

        Args:
            parameters: the current global model
            config: parameters allownig the server to influence local training process

        Returns:
            The trained local weights, the size of training data and a dictionnary of metrics
        """
        set_weights(self.net, parameters)
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters: list[float], config: dict[str, Scalar]) -> tuple[list[float], int, dict[str, Scalar]]:
        """ Test the recieved model on local data.

        Args:
            parameters: the current global model
            config: parameters allownig the server to influence local evaluation process

        Returns:
            The measured loss, the size of evaluation data and a dictionnary of metrics
        """
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context) -> FlowerClient:
    """ Instantiate FlowerClient instances.

        This function will be triggered at each round since Flower clients are stateless.
        See how to build stateful clients at <https://flower.ai/docs/framework/how-to-design-stateful-clients.html>

        Args:
            context: node context for configuration

        Returns:
            The instance of FlowerClient configured with the given context.
        """
    # Create a new model
    net = Net()

    # Get parameters from context
    partition_id = context.node_config["id"]
    num_partitions = context.node_config["num-partitions"]
    local_epochs = context.run_config["local-epochs"]
    
    # load data
    trainloader, valloader = load_data(partition_id, num_partitions)

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
