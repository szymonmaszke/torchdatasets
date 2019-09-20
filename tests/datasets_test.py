import torch

import torchdata

from .utils import enumerate_step


def test_tensor():
    dataset = torchdata.datasets.TensorDataset(
        torch.randn(100, 64), torch.randint(-5, 5, (100,))
    ).map(torchdata.maps.ToAll(lambda sample: torch.abs(sample)))
    for sample, label in dataset:
        assert torch.all(sample >= 0)
        assert label >= 0


def test_generator():
    dataset = torchdata.datasets.Generator(iter(range(50))).filter(lambda i: i % 3 == 0)
    for i, elem in enumerate_step(dataset, step=3):
        assert i == elem
