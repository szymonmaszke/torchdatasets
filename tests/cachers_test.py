import pathlib

import torch
import torchdatasets
import torchfunc

from .datasets import ExampleDataset, ExampleTensorDataset
from .utils import artificial_slowdown, index_is_sample, is_on_disk

from multiprocessing import Process, Manager

def test_pickle_cache_slowdown():
    with torchdatasets.cachers.Pickle(pathlib.Path("./disk")) as pickler:
        dataset = ExampleDataset(0, 5).map(artificial_slowdown).cache(pickler)
        with torchfunc.Timer() as timer:
            index_is_sample(dataset)
            assert timer.checkpoint() > 5
            index_is_sample(dataset)
            assert timer.checkpoint() < 0.2


def test_pickle_cache():
    datapoints = 10
    path = pathlib.Path("./disk")
    with torchdatasets.cachers.Pickle(path) as pickler:
        dataset = ExampleDataset(0, 10).cache(pickler)
        for _ in dataset:
            pass
        for i in range(datapoints):
            assert is_on_disk(path, i, ".pkl")


def test_tensor_cache():
    datapoints = 5
    path = pathlib.Path("./disk")
    with torchdatasets.cachers.Tensor(path) as cacher:
        dataset = ExampleTensorDataset(datapoints).cache(cacher)
        for _ in dataset:
            pass
        for i in range(datapoints):
            assert is_on_disk(path, i, ".pt")

def test_memory_cache():
    dataset = (
        ExampleTensorDataset(1000)
        .map(lambda tensor: tensor * 2)
        .map(lambda tensor: tensor + tensor)
        .cache(torchdatasets.cachers.Memory())
    )

    dataloader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=10)
    with torchfunc.Timer() as timer:
        for _ in dataloader:
            pass
        initial_pass = timer.checkpoint()
        for _ in dataloader:
            pass
        cached_pass = timer.checkpoint()
        assert cached_pass < initial_pass



def shared_subprocess(cache, refs):
    cacher = torchdatasets.cachers.Memory(cache)

    if id(cacher.cache) in refs.keys():
        refs[id(cacher.cache)] += 1

    dataset = (
        ExampleTensorDataset(1000)
        .map(lambda tensor: tensor * 2)
        .map(lambda tensor: tensor + tensor)
        .cache(cacher)
    )
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=10)
    with torchfunc.Timer() as timer:
        for _ in dataloader:
            pass
        initial_pass = timer.checkpoint()
        for _ in dataloader:
            pass
        cached_pass = timer.checkpoint()
        assert cached_pass < initial_pass

    assert len(cacher.cache) > 0

def test_shared_memory():
    torch.multiprocessing.set_sharing_strategy('file_system')

    manager = Manager()
    cache = manager.dict()
    refs = manager.dict()
    refs[id(cache)] = 0

    # Test speedup
    shared_subprocess(cache, refs)

    # Test shared cache with mp
    shared_cache = Manager().dict()
    del refs
    refs = manager.dict()
    refs[id(shared_cache)] = 0

    # Simulate multiprocesses training (i.e. DDP)
    procs = []
    n_processes = 2
    for i in range(n_processes):
        p = Process(target=shared_subprocess, args=(shared_cache, refs))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # Check if provided cache was used
    assert len(shared_cache) > 0

    # Check that all of the processes actually used the same cache
    assert refs[id(shared_cache)] == n_processes
