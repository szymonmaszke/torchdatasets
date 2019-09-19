import torchdata

from .datasets import ExampleDataset
from .utils import pass_through_dataset


def test_UpToIndex():
    dataset = ExampleDataset(0, 50)
    cacher = torchdata.cachers.Memory()
    dataset = dataset.cache(torchdata.modifiers.UpToIndex(10, cacher))
    pass_through_dataset(dataset)
    assert cacher.cache == {key: key for key in range(10)}


def test_FromIndex():
    dataset = ExampleDataset(0, 50)
    cacher = torchdata.cachers.Memory()
    dataset = dataset.cache(torchdata.modifiers.FromIndex(10, cacher))
    pass_through_dataset(dataset)
    assert cacher.cache == {key: key for key in range(11, len(dataset))}


def test_or():
    dataset = ExampleDataset(0, 50)
    cacher = torchdata.cachers.Memory()
    dataset = dataset.cache(
        torchdata.modifiers.UpToIndex(25, cacher)
        & torchdata.modifiers.Lambda(lambda index: index % 2 == 0, cacher)
    )
    pass_through_dataset(dataset)
    assert cacher.cache == {key: key for key in range(25) if key % 2 == 0}


def test_FromPercentage():
    dataset = ExampleDataset(0, 50)
    cacher = torchdata.cachers.Memory()
    dataset = dataset.cache(
        torchdata.modifiers.UpToPercentage(0.2, len(dataset), cacher)
    )
    pass_through_dataset(dataset)
    assert cacher.cache == {key: key for key in range(int(0.2 * len(dataset)))}
