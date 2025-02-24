from federated_learning_attack_framework.task import Net
import torch
from torchsummary import summary

net = Net()
print(net(torch.rand((1, 3, 32, 32))))
