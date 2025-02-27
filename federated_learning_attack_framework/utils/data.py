#federated-learning-attack-framework/utils/data.py

"""Provide data partitioning.

This module allows the user to partition data and get a partition.

The module contains the following functions:

- `load_data()` - Set the model state dict from a given list.
"""

from random import randint

import torch
from torch.utils.data import Dataset, random_split

from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.datasets import CIFAR10


class CustomSubset(Dataset):
    """
    Custom subset that keep the original dataset arguments

    Args:
        dataset (Dataset): The dataset to host
        indexes (list[int]): The indexes of the subset
    """

    def __init__(self, dataset, indexes):
        self.dataset = dataset
        self.indices = indexes
        self.targets = [dataset.targets[i] for i in indexes]

        # Copy attributes if exists
        if hasattr(dataset, 'classes'):
          self.classes = dataset.classes
        if hasattr(dataset, 'transform'):
          self.transform = dataset.transform
        if hasattr(dataset, 'target_transform'):
          self.target_transform = dataset.target_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        img, label = self.dataset[original_idx]
        return {'img': img, 'label': torch.tensor(label)}
    

def load_dataset() -> Dataset:
    """Load dataset
    
    This function should be adapted for different learning task
    """
    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return CIFAR10(root="./data", train=True, download=True, transform=transform)



def get_partition(partition_ratio: float) -> list[int]:
    """Get partition idx of a subset of the original dataset
    
    Args:
        partition_ratio (float): the percentage of the dataset to keep for the partition.

    Returns:
        A list with the indexes of the sampled subset.
    """
    l_dataset = len(load_dataset())
    s = set()

    while len(s) < partition_ratio*l_dataset :
        s.add(randint(1, l_dataset-1))

    return list(s)



def load_data(partition_idx: list[int], test_size=0.2) -> CustomSubset:
    """Load partition data sample with given indexes.
    
    Args:
        partition_idx (list[int]): The indexes of the subset in the dataset
        test_size (float): The percentage of test value for train/test split.

    Returns:
        A trainset and a testset sampled from the dataset with the given indexes
    """
    dataset = load_dataset()
    subset = CustomSubset(dataset, partition_idx)
    
    test_size = int(test_size * len(subset))
    train_size = len(subset) - test_size
    
    trainset, testset = random_split(subset, [train_size, test_size])
    
    return trainset, testset
