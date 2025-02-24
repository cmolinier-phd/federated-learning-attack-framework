"""Generic federated learning attack framework

This package gives a baseline to experiment attacks and defense against federated learning.

It's build on top of flower framework. see more at https://flower.ai/.

Modules exported by this package:
- `client_app`: Provides client app implementation with benign clients and adversaries
- `server_app`: Provides server app implementation with selected strategies
- `task`: Provides model implementation, data loading pipeline, training and testing functions and model weights getter/setter
"""
