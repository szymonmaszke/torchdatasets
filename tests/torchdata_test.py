import time

import torch

import torchdata
import torchfunc

from .datasets import ExampleDataset, ExampleIterable
from .utils import artificial_slowdown, enumerate_step, index_is_sample


def test_basic_iterable():
    dataset = ExampleIterable(0, 100).map(lambda value: value + 12)
    for index, item in enumerate(dataset):
        assert index + 12 == item


def test_iterable_filter():
    dataset = (
        ExampleIterable(0, 100)
        .map(lambda value: value + 12)
        .filter(lambda elem: elem % 2 == 0)
    )
    for index, item in enumerate_step(dataset, start=12, step=2):
        assert index == item


def test_basic_dataset():
    dataset = ExampleDataset(0, 25).map(lambda sample: sample * sample).cache()
    for index, value in enumerate(dataset):
        assert index ** 2 == value


def test_dataset_multiple_cache():
    # Range-like Dataset mapped to item ** 3
    dataset = (
        ExampleDataset(0, 25)
        .cache()
        .map(lambda sample: (sample + sample, sample))
        .cache()
        .map(lambda sample: sample[0] - sample[-1])
        .cache()
        .map(lambda sample: sample ** 3)
        .cache()
    )
    # Iterate through dataset
    for _ in dataset:
        pass

    for index, value in enumerate(dataset):
        assert index ** 3 == value


def test_dataset_cache_speedup():
    dataset = ExampleDataset(0, 5).map(artificial_slowdown).cache()
    with torchfunc.Timer() as timer:
        index_is_sample(dataset)
        assert timer.checkpoint() > 5
        index_is_sample(dataset)
        assert timer.checkpoint() < 0.2


def test_dataset_complicated_cache():
    dataset = (
        (
            (
                ExampleDataset(0, 25)
                | ExampleDataset(0, 25).map(lambda value: value * -1)
            )
            .cache()
            .map(lambda sample: sample[0] + sample[1] + sample[0])
            .cache()
            .map(lambda sample: sample + sample)
            | ExampleDataset(0, 25)
        )
        .cache()
        .map(lambda values: ((values, values), values))
        .map(torchdata.maps.Flatten())
        .cache()
        .map(lambda values: values[1])
        .map(lambda value: value ** 2)
    )

    for index, value in enumerate(dataset):
        assert index ** 2 == value


def test_apply():
    def summation(generator):
        return sum(value for value in generator)

    assert ExampleDataset(0, 101).apply(summation) == 5050  # Returns 5050


def test_reduce():
    assert ExampleDataset(0, 10).reduce(lambda x, y: x + y) == 45


def test_reduce_initializer():
    assert ExampleDataset(0, 10).reduce(lambda x, y: x + y, 10) == 55


def test_repr():
    assert (
        repr(ExampleDataset(0, 5))
        == "tests.datasets.ExampleDataset(values=[0, 1, 2, 3, 4])"
    )


def test_dataset_dataloader():
    # Range-like Dataset mapped to item ** 3
    dataset = (
        ExampleDataset(0, 25)
        .cache()
        .map(lambda sample: (sample + sample, sample))
        .cache()
        .map(lambda sample: sample[0] - sample[-1])
        .cache()
        .map(lambda sample: sample ** 3)
        .cache()
    )
    # Iterate through dataset
    for element in torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=3):
        print(element)
