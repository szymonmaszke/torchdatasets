import typing

import torchdatasets

from .datasets import ExampleDataset
from .utils import create_dataset_many_samples, index_is_sample, is_none


def test_after():
    dataset = ExampleDataset(0, 25).map(torchdatasets.maps.After(10, lambda x: x ** 2))
    for index, element in enumerate(dataset):
        if index > 10:
            assert element == index ** 2
        else:
            assert element == index


def test_onsignal():
    class Handle:
        def __init__(self):
            self.value: bool = False

        def __call__(self):
            return self.value

    handler = Handle()
    dataset = ExampleDataset(0, 25).map(
        torchdatasets.maps.OnSignal(handler, lambda x: x ** 2)
    )
    for index, element in enumerate(dataset):
        if index == 10:
            handler.value = True
        if index <= 10:
            assert element == index
        else:
            assert element == index ** 2


def test_repeat():
    dataset = ExampleDataset(0, 25).map(torchdatasets.maps.Repeat(10, lambda x: x + 1))
    index_is_sample(dataset, modifier=lambda x: x + 10)


def test_select():
    index_is_sample(create_dataset_many_samples(2).map(torchdatasets.maps.Select(0)))


def test_select_none():
    is_none(create_dataset_many_samples(2).map(torchdatasets.maps.Select()))


def test_select_multiple():
    index_is_sample(
        create_dataset_many_samples(3)
        .map(torchdatasets.maps.Select(0, 1))
        .map(torchdatasets.maps.Select(1))
    )


def test_flatten_flattened():
    index_is_sample(create_dataset_many_samples(1).map(torchdatasets.maps.Flatten()))


def test_flatten_signle():
    index_is_sample(ExampleDataset(0, 10).map(torchdatasets.maps.Flatten()))


def test_drop():
    index_is_sample(create_dataset_many_samples(2).map(torchdatasets.maps.Drop(0)))


def test_drop_all():
    is_none(create_dataset_many_samples(2).map(torchdatasets.maps.Drop(0, 1)))


def test_drop_multiple():
    index_is_sample(
        create_dataset_many_samples(3)
        .map(torchdatasets.maps.Drop(0, 2))
        .map(torchdatasets.maps.Select(1))
    )


def test_to_all():
    dataset = create_dataset_many_samples(8).map(torchdatasets.maps.ToAll(lambda x: -x))
    for index, sample in enumerate(dataset):
        for subsample in sample:
            assert -index == subsample


def indices_test(to: bool, length: int, threshold: int):
    def modifier(subsample):
        if to:
            return -subsample
        return subsample

    dataset = create_dataset_many_samples(length).map(
        (torchdatasets.maps.To if to else torchdatasets.maps.Except)(
            lambda x: -x, *list(range(threshold))
        )
    )

    for index, sample in enumerate(dataset):
        for subsample in sample[:threshold]:
            assert index == modifier(subsample)
        for subsample in sample[threshold:]:
            assert -index == modifier(subsample)


def test_to():
    indices_test(True, length=8, threshold=5)


def test_except():
    indices_test(False, length=8, threshold=5)
