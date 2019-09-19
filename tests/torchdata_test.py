import time

import torchdata
import torchfunc

from .datasets import ExampleDataset, ExampleIterable


def test_basic_iterable():
    dataset = ExampleIterable(0, 100).map(lambda value: value + 12)
    for index, item in enumerate(dataset):
        assert index + 12 == item


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
    def artificial_slowdown(sample):
        time.sleep(1)
        return sample

    def pass_through_dataset(dataset):
        for correct, sample in enumerate(dataset):
            assert correct == sample

    dataset = ExampleDataset(0, 5).map(artificial_slowdown).cache()
    with torchfunc.Timer() as timer:
        pass_through_dataset(dataset)
        assert timer.checkpoint() > 5
        pass_through_dataset(dataset)
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
