import torch

import torchdatasets
import torchvision

from .utils import enumerate_step


def test_tensor():
    dataset = torchdatasets.datasets.TensorDataset(
        torch.randn(100, 64), torch.randint(-5, 5, (100,))
    ).map(torchdatasets.maps.ToAll(torch.abs))
    for sample, label in dataset:
        assert torch.all(sample >= 0)
        assert label >= 0


def test_generator():
    dataset = torchdatasets.datasets.Generator(iter(range(50))).filter(lambda i: i % 3 == 0)
    for i, elem in enumerate_step(dataset, step=3):
        assert i == elem


def test_wrapper():
    dataset = torchdatasets.datasets.WrapDataset(
        torchvision.datasets.MNIST("./data", download=True, train=True)
    ).map(lambda sample: (sample[0], 0))
    for _, label in dataset:
        assert label == 0
    assert dataset.train
