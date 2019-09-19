import pathlib

import torchdata
import torchfunc

from .datasets import ExampleDataset
from .utils import artificial_slowdown, index_is_sample


def test_disk_cache():
    with torchdata.cachers.Pickle(pathlib.Path("./disk")) as pickler:
        dataset = ExampleDataset(0, 5).map(artificial_slowdown).cache(pickler)
        with torchfunc.Timer() as timer:
            index_is_sample(dataset)
            assert timer.checkpoint() > 5
            index_is_sample(dataset)
            assert timer.checkpoint() < 0.2
