#federated-learning-attack-framework/client.py

"""Provide client app.

This module allows the user to instantiate clients and launch them in the federation.

The module contains the following functions:

- `FlowerClient()` - The client's class.
- `client_fn(context)` - The function to instantiate clients.
"""

from numpy.random import poisson

import torch
from torch import nn as nn
from torch.utils.data import DataLoader

from flwr.common import Context, ConfigsRecord
from flwr.common.typing import Scalar
from flwr.client import ClientApp, NumPyClient

from federated_learning_attack_framework.utils.task import Net, get_weights, set_weights, test, train
from federated_learning_attack_framework.utils.data import load_data, get_partition


class FlowerClient(NumPyClient):
    """Simple client from NumpyClient class

    For each round where the client is triggered, it will follow these steps:

    1. Replace its model with the received weights
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
        """ Train the received model on local data.

        Args:
            parameters: the current global model
            config: parameters allowing the server to influence local training process

        Returns:
            The trained local weights, the size of training data and a dictionary of metrics
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
        """ Test the received model on local data.

        Args:
            parameters: the current global model
            config: parameters allowing the server to influence local evaluation process

        Returns:
            The measured loss, the size of evaluation data and a dictionary of metrics
        """
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


class DummyAttacker(FlowerClient):
    """Simple attacker from NumpyClient class

    For each round where the attacker is triggered, it will behaves the same as a benign client (no attack)
    """
    def __init__(self, net, trainloader, valloader, local_epochs):
        super().__init__(net, trainloader, valloader, local_epochs)

    def fit(self, parameters: list[float], config: dict[str, Scalar]) -> tuple[list[float], int, dict[str, Scalar]]:
        return super().fit(parameters, config)
    
    def evaluate(self, parameters: list[float], config: dict[str, Scalar]) -> tuple[list[float], int, dict[str, Scalar]]:
        return super().evaluate(parameters, config)
    
    
def client_fn(context: Context) -> FlowerClient:
    """ Instantiate FlowerClient instances.

        This function will be triggered at each round since Flower clients are stateless.
        See how to build stateful clients at <https://flower.ai/docs/framework/how-to-design-stateful-clients.html>

        To setup the adversaries, this function will use the run config parameter `n_attackers` and will instantiate all clients with id under num_attackers as attackers

        Args:
            context: node context for configuration

        Returns:
            The instance of FlowerClient configured with the given context.
        """
    # Create a new model
    net = Net()

    # Get parameters from context
    id = context.node_config["partition-id"]
    local_epochs = context.run_config["local-epochs"]

    if "partition_config" not in context.state.configs_records:
        partition_ratio = poisson(7)/28 # Obtained via an empirical study
        context.state.configs_records["partition_config"] = ConfigsRecord()
        context.state.configs_records["partition_config"]['partition_idx'] = get_partition(partition_ratio)
    
    # load data
    trainset, testset = load_data(context.state.configs_records["partition_config"]['partition_idx'])

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32)

    # Return Client instance
    return FlowerClient(net, trainloader, testloader, local_epochs).to_client() if id > context.run_config["n_attackers"] else DummyAttacker(net, trainloader, testloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
