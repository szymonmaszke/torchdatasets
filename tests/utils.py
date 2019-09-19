import time
import typing


def artificial_slowdown(sample):
    time.sleep(1)
    return sample


def index_is_sample(dataset, modifier: typing.Callable = None):
    if modifier is None:
        modifier = lambda x: x
    for correct, sample in enumerate(dataset):
        assert correct == modifier(sample)
