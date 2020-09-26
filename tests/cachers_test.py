import pathlib

import torch
import torchdata
import torchfunc

from .datasets import ExampleDataset, ExampleTensorDataset
from .utils import artificial_slowdown, index_is_sample, is_on_disk


def test_pickle_cache_slowdown():
    with torchdata.cachers.Pickle(pathlib.Path("./disk")) as pickler:
        dataset = ExampleDataset(0, 5).map(artificial_slowdown).cache(pickler)
        with torchfunc.Timer() as timer:
            index_is_sample(dataset)
            assert timer.checkpoint() > 5
            index_is_sample(dataset)
            assert timer.checkpoint() < 0.2


def test_pickle_cache():
    datapoints = 10
    path = pathlib.Path("./disk")
    with torchdata.cachers.Pickle(path) as pickler:
        dataset = ExampleDataset(0, 10).cache(pickler)
        for _ in dataset:
            pass
        for i in range(datapoints):
            assert is_on_disk(path, i, ".pkl")


def test_tensor_cache():
    datapoints = 5
    path = pathlib.Path("./disk")
    with torchdata.cachers.Tensor(path) as cacher:
        dataset = ExampleTensorDataset(datapoints).cache(cacher)
        for _ in dataset:
            pass
        for i in range(datapoints):
            assert is_on_disk(path, i, ".pt")


def test_shared_memory():
    dataset = (
        ExampleTensorDataset(1000)
        .map(lambda tensor: tensor * 2)
        .map(lambda tensor: tensor + tensor)
        .cache(torchdata.cachers.SharedMemory())
    )
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=2, batch_size=10)
    with torchfunc.Timer() as timer:
        for _ in dataloader:
            pass
        initial_pass = timer.checkpoint()
        for _ in dataloader:
            pass
        cached_pass = timer.checkpoint()
        assert cached_pass < initial_pass
