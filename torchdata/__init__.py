r"""**This module contains PyTorch compatible datasets with extended capabilities.**

To quickly start with `torchdata`, just inherit from `torchdata.Dataset` and create
your dataset as you normally would, for example::

    import torchvision
    from PIL import Image

    # Image loading dataset (use torchdata.Files for even less typing :D )
    class Dataset(torchdata.Dataset):
        def __init__(self, path: pathlib.Path):
            super().__init__() # This is necessary
            self.files = [file for file in path.glob("*")]

        def __getitem__(self, index):
            return Image.open(self.files[index])

        def __len__(self, index):
            return len(self.files)


Now you can use `cache`, `map` and `apply` just by issuing appropriate functions::

    # Map PIL to Tensor and cache dataset
    dataset = Dataset("data").map(torchvision.transforms.ToTensor()).cache()
    # You can create DataLoader as well
    dataloader = torch.utils.data.DataLoader(dataset)

For custom caching routines and how to use them see `cachers`, to check available
general `maps` see `maps`. `modifiers` can further modify `cachers`, you should
see it's respective docs as well.

"""

import abc
import pathlib
import typing

from torch.utils.data import ChainDataset as TorchChain
from torch.utils.data import ConcatDataset as TorchConcatDataset
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import IterableDataset as TorchIterable
from torch.utils.data import TensorDataset as TorchTensorDataset

from . import cachers, maps, samplers
from ._base import Base, MetaDataset, MetaIterable


class _DatasetBase(Base):
    def __init__(self, concat_object, chain_object):
        self._maps = []
        self._concat_object = concat_object
        self._chain_object = chain_object

    def map(self, function: typing.Callable):
        r"""**Apply function to each element of dataset.**

        Function has no specified signature; it is user's responsibility to ensure
        it is taking correct arguments as returned from `__getitem__`.

        Parameters
        ----------
        function: typing.Callable
                Function (or functor) taking arguments returned from `__getitem__`
                and returning anything.

        Returns
        -------
        self

        """
        self._maps.append(function)
        return self

    def __or__(self, other):
        rf"""**Concatenate {self} and another {self} compatible object.**

        During iteration, items from both dataset will be returned as `tuple`.
        Another object could be PyTorch's base class of this object.

        Length of resulting dataset is equal to `min(len(self), len(other))`

        Parameters
        ----------
        other : {self} or PyTorch's base counterpart
                Dataset instance whose sample will be iterated over together

        Returns
        -------
        {self._concat_object}
                Proxy object responsible for concatenation between samples.
                Can be used in the same manner as this object.

        """
        return self._concat_object((self, other))

    def __add__(self, other):
        rf"""**Chain {self} and another {self} compatible object.**

        During iteration, items from self will be returned first and items
        from other dataset after those.

        Length of such dataset is equal to `len(self) + len(other)`

        Parameters
        ----------
        other : {self} or PyTorch's base counterpart
                Dataset whose sample will be yielded after this dataset.

        Returns
        -------
        {self._chain_object}
                Proxy object responsible for chaining datasets.
                Can be used in the same manner as this object.

        """
        return self._chain_object((self, other))


class Iterable(TorchIterable, _DatasetBase, metaclass=MetaIterable):
    r"""`torch.utils.data.IterableDataset` **dataset with** `map` **capabilities**.

    This class inherits from `torch.utils.data.Iterable`, so can be used exactly
    the same. To get it's capabilities, inherit from this class, see example below.

    It allows user to perform following operations:

    - `map` - apply function to each element of dataset

    **Example**::

        # Based on original PyTorch example
        class Dataset(torchdata.Iterable):
            def __init__(self, start: int, end: int):
                super().__init__() # This is necessary
                self.start: int = start
                self.end: int = end

            def __iter__(self):
                return iter(range(self.start, self.end))

        # range(1,25) originally, mapped to range(13, 37)
        dataset = Dataset(1, 25).map(lambda value: value + 12)
        # Sample-wise concatenation, yields range(13, 37) and range(1, 25)
        for first, second in dataset | Dataset(1, 25):
            print(first, second) # 13 1 up to 37 25

    """

    def __init__(self):
        _DatasetBase.__init__(self, ConcatIterable, ChainIterable)


class Dataset(TorchDataset, _DatasetBase, metaclass=MetaDataset):
    r"""`torch.utils.data.Dataset` **with** `map`**,** `cache` **and** `apply` **support**

    This class inherits from `torch.utils.data.Dataset`, so can be used exactly
    the same. To get it's capabilities, inherit from this class, see example below.

    It allows user to perform the following operations:

    - `cache` - cache all/part of data in memory or on disk
    - `map` - apply function to each element of dataset
    - `apply` - apply function to **all** elements of dataset

    **Important:** Last cache which is able to hold sample is used.
    Does not matter whether it's in-memory or on-disk or user-specified.

    **Important:** Although multiple cache calls in different parts of `map`
    should work, users are encouraged to use it as rare as possible and possibly
    as late as possible for best performance.

    **Example**::

        import torchvision
        from PIL import Image

        # Image loading dataset (use Files for more serious business)
        class Dataset(torchdata.Dataset):
            def __init__(self, path: pathlib.Path):
                super().__init__() # This is necessary
                self.files = [file for file in path.glob("*")]

            def __getitem__(self, index):
                return Image.open(self.files[index])

            def __len__(self, index):
                return len(self.files)

        # Map PIL to Tensor and cache dataset
        dataset = Dataset("data").map(torchvision.transforms.ToTensor()).cache()
        # Create DataLoader as normally
        dataloader = torch.utils.data.DataLoader(dataset)

    """

    def __init__(self):
        _DatasetBase.__init__(self, ConcatDataset, ConcatIterable)
        self._cachers = []
        self._which = []

    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def __getitem__(self, index):
        pass

    def apply(self, function: typing.Callable, *args, **kwargs):
        r"""**Apply function to every element of the dataset.**

        Specified function has to take Python generator as first argument.
        This generator yields consecutive samples from the dataset and the function is free
        to do whatever it wants with them.

        Other arguments will be forwarded to function.

        **WARNING:**

        This function returns anything that's returned from function
        and it's up to user to ensure correct pipeline functioning
        after using this transformation.

        **Example**::

            class Dataset(torchdata.Dataset):
                def __init__(self, max: int):
                    super().__init__() # This is necessary
                    self.range = list(range(max))

                def __getitem__(self, index):
                    return self.range[index]

                def __len__(self):
                    return len(self.range)

            def summation(generator):
                return sum(value for value in generator)

            summed_dataset = Dataset(101).apply(summation) # Returns 5050


        Parameters
        ----------
        function : typing.Callable
                Function (or functional object) taking item generator as first object
                and variable list of other arguments (if necessary).

        Returns
        -------
        typing.Any
                Value returned by function

        """
        return function((value for value in self), *args, **kwargs)

    def cache(self, cacher: typing.Callable = None):
        r"""**Cache data in memory, disk or specify custom caching.**

        By default all samples are cached in memory. To change this behaviour specify `cacher`
        argument. Some `cacher` implementations can be found in `torchdata.cacher` module or you can
        provide your own by inheriting from `torchdata.cacher.Cacher` and implementing
        appropriate methods.

        Parameters
        ----------
        cacher : torchdata.cacher.Cacher, optional
                Instance of `torchdata.cacher.Cacher` (or any other object with compatible interface).
                Check `cacher` module documentation for more information.
                Default: `torchdata.cacher.Memory` which caches data in-memory

        Returns
        -------
        Dataset
                Returns self

        """
        if cacher is None:
            cacher = cachers.Memory()
        self._cachers.append(cacher)
        self._which.append(len(self._maps))
        return self

    def reset(self, cache: bool = True, maps: bool = True):
        r"""**Reset dataset state.**

        `cache` and `maps` can be resetted separately.

        Parameters
        ----------
        cache : bool, optional
                Reset current cache. Default: True
        maps : bool, optional
                Reset current disk cache. Default: True

        """

        if cache:
            self._cachers = []
        if maps:
            self._maps = []


################################################################################
#
#                           Dataset Concatenations
#
################################################################################


class ConcatDataset(Dataset):
    r"""**Concrete** `torchdata.Dataset` **responsible for sample-wise concatenation.**

    This class is returned when `|` (logical or operator) is used on instance
    of `torchdata.Dataset` (original `torch.utils.data.Dataset` can be used as well).

    Behaves the same as `torchdata.Dataset` otherwise.

    **Important:** This class is meant to be more of a proxy for `|` operator,
    you can use it directly though.

    **Example**::

        dataset = (
            torchdata.ConcatDataset([dataset1, dataset2, dataset3])
            .map(lambda x, y, z: (x + y, z))
            .cache()
        )

    Attributes
    ----------
    datasets : List[Union[torchdata.Dataset, torch.utils.data.Dataset]]
            List of datasets to be concatenated sample-wise.

    """

    def __init__(self, datasets: typing.List):
        super().__init__()
        self.datasets = datasets

    def __getitem__(self, index):
        return tuple(dataset[index] for dataset in self.datasets)

    def __len__(self):
        return min(len(dataset) for dataset in self.datasets)


class ConcatIterable(Iterable):
    r"""**Concrete** `torchdata.Iterable` **responsible for sample-wise concatenation.**

    This class is returned when `|` (logical or operator) is used on instance
    of `torchdata.Iterable` (original `torch.utils.data.IterableDataset` can be used as well).

    Behaves the same as `torchdata.Iterable` otherwise.

    **Important:** This class is meant to be more of a proxy for `|` operator,
    you can use it directly though.

    **Example**::

        dataset = (
            torchdata.ConcatIterable([dataset1, dataset2, dataset3])
            .map(lambda x, y, z: (x + y, z))
            .cache()
        )

    Attributes
    ----------
    datasets : List[Union[torchdata.Iterable, torch.utils.data.IterableDataset]]
            List of datasets to be concatenated sample-wise.

    """

    def __init__(self, datasets: typing.List):
        super().__init__()
        self.datasets = datasets

    def __iter__(self):
        yield from zip(*self.datasets)

    def __getitem__(self, index):
        return tuple(dataset[index] for dataset in self.datasets)

    def __len__(self):
        return min(len(dataset) for dataset in self.datasets)


class ChainDataset(TorchConcatDataset, Dataset):
    r"""**Concrete** `torchdata.Dataset` **responsible for chaining multiple datasets.**

    This class is returned when `+` (logical or operator) is used on instance
    of `torchdata.Dataset` (original `torch.utils.data.Dataset` can be used as well).
    Acts just like PyTorch's `+` and `ConcatDataset`, see https://pytorch.org/docs/stable/data.html

    Behaves the same as `torchdata.Dataset` otherwise.

    **Important:** This class is meant to be more of a proxy for `+` operator,
    you can use it directly though.

    **Example**::

        # Iterate over 3 datasets consecutively
        dataset = (
            torchdata.ChainDataset([dataset1, dataset2, dataset3])
            .map(lambda x: x**2)
            .cache()
        )

    Attributes
    ----------
    datasets : List[Union[torchdata.Dataset, torch.utils.data.Dataset]]
            List of datasets to be chained.

    """

    def __init__(self, datasets):
        Dataset.__init__(self)
        TorchConcatDataset.__init__(self, datasets)


class ChainIterable(TorchChain, Iterable):
    r"""**Concrete** `torchdata.Iterable` **responsible for chaining multiple datasets.**

    This class is returned when `+` (logical or operator) is used on instance
    of `torchdata.Iterable` (original `torch.utils.data.Iterable` can be used as well).
    Acts just like PyTorch's `+` and `ChainDataset`, see https://pytorch.org/docs/stable/data.html

    Behaves the same as `torchdata.Iterable` otherwise.

    **Important:** This class is meant to be more of a proxy for `+` operator,
    you can use it directly though.

    **Example**::

        # Iterate over 3 iterable datasets consecutively
        dataset = (
            torchdata.ChainDataset([dataset1, dataset2, dataset3])
            .map(lambda x: x**2)
            .cache()
        )

    Attributes
    ----------
    datasets : List[Union[torchdata.Iterable, torch.utils.data.IterableDataset]]
            List of datasets to be chained.

    """

    def __init__(self, datasets):
        Iterable.__init__(self)
        TorchChain.__init__(self, datasets)


################################################################################
#
#                           GENERAL PURPOSE ABSTRACT
#
################################################################################


class FilesDataset(Dataset):
    r"""**Create** `Dataset` **from list of files.**

    Each file is a separate sample. After inheritance, user has to specify
    `__getitem__` method responsible for file indexing.

    **Example**::

        import torchdata
        import torchvision

        from PIL import Image


        # Image loading dataset
        class ImageDataset(torchdata.FilesDataset):
            def __getitem__(self, index):
                return Image.open(self.files[index])

        # Useful class methods are inherited as well
        dataset = ImageDataset.from_folder("./data", regex="*.png").map(
            torchvision.transforms.ToTensor()
        )

    `from_folder` class method is available for common case of creating dataset
    from files in folder.

    Parameters
    ----------
    files : List[pathlib.Path]
            List of files to be used.
    regex : str, optional
            Regex to be used  for filtering. Default: `*` (all files)
    *args
            Arguments saved for `__iter__`
    **kwargs
            Keyword arguments saved for `__iter__`

    """

    @classmethod
    def from_folder(cls, path: pathlib.Path, regex: str = "*", *args, **kwargs):
        r"""**Create dataset from** `pathlib.Path`**-like object.**

        Path should be a directory and will be extended via `glob` method taking `regex`
        (if specified). Varargs and kwargs will be saved for use for `__getitem__` method.

        Parameters
        ----------
        path : pathlib.Path
                Path object (directory) containing samples.
        regex : str, optional
                Regex to be used  for filtering. Default: `*` (all files)
        *args
                Arguments saved for `__getitem__`
        **kwargs
                Keyword arguments saved for `__getitem__`

        Returns
        -------
        FilesDataset
                Instance of your file based dataset.
        """

        files = [file for file in path.glob(regex)]
        return cls(files, *args, **kwargs)

    def __init__(self, files: typing.List[pathlib.Path], *args, **kwargs):
        super().__init__()
        self.files: typing.List = files
        self.args = args
        self.kwargs = kwargs

    def __len__(self):
        return len(self.files)

    def filter(self, predicate: typing.Callable):
        r"""**Remove files for which predicate returns** `False`**.**

        Parameters
        ----------
        predicate : Callable
                Function-like object taking file as argument and returning boolean
                indicating whether to keep a file.

        Returns
        -------
        FilesDataset
                Modified self
        """
        self.files = [file for file in self.files if predicate(file)]
        return self

    def sort(self, key=None, reverse=False):
        r"""**Sort files using Python's built-in** `sorted` **method.**

        Arguments are passed directly to `sorted`.

        Parameters
        ----------
        key: Callable, optional
            Specifies a function of one argument that is used to extract a comparison key from each element.
            Default: `None` (compare the elements directly).

        reverse: bool, optional
            Whether `sorting` should be descending. Default: `False`

        Returns
        -------
        FilesDataset
                Modified self

        """
        self.files = sorted(self.files, key=key, reverse=reverse)
        return self


class FilesIterable(Iterable):
    r"""**Create** `Iterable` **from folder.**

    Should be used instead of FilesDataset when there is too many files in the folder
    to be kept in the memory.

    Each file is a separate sample. After inheritance, user has to specify
    `__iter__` method responsible for iterating over files.

    **Example**::

        class SimpleFilesIterable(FilesIterable):
            def __iter__(self):
                for path in self.path.glob(self.regex):
                    with open(path) as file:
                        yield file

    Parameters
    ----------
    path: pathlib.Path
            Path object (directory) containing samples.
    regex: str, optional
            Regex to be used  for filtering. Default: `*` (all files)
    *args
            Arguments saved for `__getitem__`
    **kwargs
            Keyword arguments saved for `__getitem__`

    """

    def __init__(self, path: pathlib.Path, regex: str = "*", *args, **kwargs):
        self.path: pathlib.Path = path
        self.regex: str = regex
        self.args = args
        self.kwargs = kwargs


################################################################################
#
#                           GENERAL PURPOSE CONCRETE
#
################################################################################


class TensorDataset(TorchTensorDataset, Dataset):
    r"""**Dataset wrapping** `torch.tensors`**.**

    `cache`, `map` etc. enabled version of `torch.utils.data.TensorDataset`

    Parameters:
    -----------
    *tensors : torch.Tensor
            List of `tensors` to be wrapped.
    """

    def __init__(self, *tensors):
        Dataset.__init__(self)
        TorchTensorDataset.__init__(self, *tensors)


class Generator(Iterable):
    r"""**Iterable wrapping any generator expression.**

    Parameters:
    -----------
    *expression: Generator expression
            Generator from which one can `yield` via `yield from` syntax.
    """

    def __init__(self, expression):
        super().__init__()
        self.expression = expression

    def __iter__(self):
        yield from self.expression
