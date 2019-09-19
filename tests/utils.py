import time
import typing

import torchdata

from .datasets import ExampleDataset


def pass_through_dataset(dataset):
    for elem in dataset:
        pass


def artificial_slowdown(sample):
    time.sleep(1)
    return sample


def index_is_sample(dataset, modifier: typing.Callable = None):
    if modifier is None:
        modifier = lambda x: x
    for index, sample in enumerate(dataset):
        print(sample)
        assert modifier(index) == sample


def create_dataset_many_samples(samples):
    dataset = ExampleDataset(0, 25)
    for _ in range(samples - 1):
        dataset |= dataset
    return dataset.map(torchdata.maps.Flatten())


def is_none(dataset):
    for elem in dataset:
        assert elem is None
