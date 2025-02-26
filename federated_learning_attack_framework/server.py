#federated-learning-attack-framework/server.py

"""Provide server app.

This module allows the user to instantiate server.

The module contains the following functions:

- `server_fn(context)` - The function to instantiate the server.
"""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from federated_learning_attack_framework.utils.task import Net, get_weights


def server_fn(context: Context) -> ServerAppComponents:
    """ Instantiate server instance.

        Args:
            context: node context for configuration

        Returns:
            The instance of ServerAppComponent configured with the given context.
        """
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
