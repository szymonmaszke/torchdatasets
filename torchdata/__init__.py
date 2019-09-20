r"""**This module contains PyTorch compatible datasets with extended capabilities.**

To quickly start with `torchdata`, just inherit from `torchdata.Dataset` and create
your dataset as you normally would, for example::

    import torchdata
    from PIL import Image

    # Image loading dataset (use torchdata.Files for even less typing :D )
    class Dataset(torchdata.Dataset):
        def __init__(self, path: pathlib.Path):
            super().__init__() # This is necessary
            self.files = [file for file in path.glob("*")]

        def __getitem__(self, index):
            return Image.open(self.files[index])

        def __len__(self):
            return len(self.files)


Now you can use `cache`, `map`, `apply`, `reduce` just by issuing appropriate functions
(standard `torch.utils.data.Dataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`__
can still be used)::

    import torchvision

    # Map PIL to Tensor and cache dataset
    dataset = Dataset("data").map(torchvision.transforms.ToTensor()).cache()
    # You can create DataLoader as well
    dataloader = torch.utils.data.DataLoader(dataset)

`torchdata.Iterable` is an extension of
`torch.utils.data.IterableDataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset>`__,
which allows the user to use `map`, `apply` and `filter`, for example::

    # Based on original PyTorch example
    class Dataset(torchdata.Iterable):
        def __init__(self, start: int, end: int):
            super().__init__() # This is necessary
            self.start: int = start
            self.end: int = end

        def __iter__(self):
            return iter(range(self.start, self.end))

    # Only elements divisible by 2
    dataset = Dataset(100).filter(lambda value: value % 2 == 0)

Concrete implementations of datasets described above is located inside `datasets` module.

For custom caching routines and how to use them see `cachers` and their `modifiers`.
To check available general `map` related functions see `maps`.

Custom sampling techniques useful with `torch.utils.data.DataLoader <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`__
are located inside `samplers`.

"""

from . import cachers, datasets, maps, modifiers, samplers
from .datasets import Dataset, Iterable
