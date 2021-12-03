"""**Concrete implementations of** `torchdatasets.Dataset` **and** `torchdatasets.Iterable`.

Classes below extend and/or make it easier for user to implement common functionalities.
To use standard PyTorch datasets defined by, for example, `torchvision`, you can
use `WrapDataset` or `WrapIterable` like this::

    import torchdatasets
    import torchvision

    dataset = torchdatasets.datasets.WrapDataset(
        torchvision.datasets.MNIST("./data", download=True)
    )

After that you can use `map`, `apply` and other functionalities like you normally would with
either `torchdatasets.Dataset` or `torchdatasets.Iterable`.
"""

import abc
import functools
import pathlib
import typing

from torch.utils.data import ChainDataset as TorchChain
from torch.utils.data import ConcatDataset as TorchConcatDataset
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import IterableDataset as TorchIterable
from torch.utils.data import TensorDataset as TorchTensorDataset

from ._base import Base, MetaDataset, MetaIterable
from .cachers import Memory

try:
    from typing import GenericMeta
except ImportError:
    # in python > 3.7, genericmeta doesn't exist
    class GenericMeta(type): pass

try:
    from torch.utils.data import _typing    
    class MetaIterableWrapper(MetaIterable, GenericMeta, _typing._DataPipeMeta): pass
except ImportError:
    # for pytorch < 1.9 _typing does not exist
    class MetaIterableWrapper(MetaIterable): pass


class _DatasetBase(Base):
    def __init__(self, concat_object, chain_object):
        self._maps = []
        self._concat_object = concat_object
        self._chain_object = chain_object

    def map(self, function: typing.Callable):
        r"""**Map function to each element of dataset.**

        Function has no specified signature; it is user's responsibility to ensure
        it is taking correct arguments as returned from `__getitem__` (in case of `Dataset`)
        or `__iter__` (in case of `Iterable`).

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

    def apply(self, function):
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

            class Dataset(torchdatasets.Dataset):
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
        return function((value for value in self))

    def __or__(self, other):
        r"""**Concatenate {self} and another {self} compatible object.**

        During iteration, items from both dataset will be returned as `tuple`.
        Another object could be PyTorch's base class of this object.

        Length of resulting dataset is equal to `min(len(self), len(other))`

        Parameters
        ----------
        other : {self} or PyTorch's base counterpart
                Dataset instance whose sample will be iterated over together

        Returns
        -------
        {concat_object}
                Proxy object responsible for concatenation between samples.
                Can be used in the same manner as this object.

        """.format(
            self=self, concat_object=self._concat_object
        )
        return self._concat_object((self, other))

    def __add__(self, other):
        r"""**Chain {self} and another {self} compatible object.**

        During iteration, items from self will be returned first and items
        from other dataset after those.

        Length of such dataset is equal to `len(self) + len(other)`

        Parameters
        ----------
        other : {self} or PyTorch's base counterpart
                Dataset whose sample will be yielded after this dataset.

        Returns
        -------
        {chain_object}
                Proxy object responsible for chaining datasets.
                Can be used in the same manner as this object.

        """.format(
            self=self, chain_object=self._chain_object
        )
        return self._chain_object((self, other))


class Iterable(TorchIterable, _DatasetBase, metaclass=MetaIterableWrapper):
    r"""`torch.utils.data.IterableDataset` **dataset with extended capabilities**.

    This class inherits from
    `torch.utils.data.IterableDataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset>`__,
    co can be used in the same manner after inheritance.

    It allows user to perform following operations:

    - `map` - apply function to each element of dataset
    - `apply` - apply function to **all** elements of dataset
    - `filter` - return elements for which `predicate` returns `True`

    **Example**::

        # Based on original PyTorch example
        class Dataset(torchdatasets.Iterable):
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

    @abc.abstractmethod
    def __iter__(self):
        pass

    def __init__(self):
        _DatasetBase.__init__(self, ConcatIterable, ChainIterable)
        self._filters = []
        self._which = [0]

    def filter(self, predicate: typing.Callable):
        r"""**Filtered  data according to** `predicate`.

        Values are filtered based on value returned after every operation (including `map`)
        specified before `filter`, for example::

            dataset = (
                    ExampleIterable(0, 100)
                    .map(lambda value: value + 50)
                    .filter(lambda elem: elem % 2 == 0)
            )

        Above will return elements `[50, 100]` divisible by `2`.

        Parameters
        ----------
        predicate: Callable -> bool
                Function returning bool and taking single argument (which is
                whatever is returned from the dataset when `filter` is applied).
                If `True`, sample will be returned, otherwise it is skipped.

        Returns
        -------
        Dataset
                Returns self

        """
        self._which.append(len(self._maps))
        self._filters.append(predicate)
        return self


class MetaDatasetWrapper(MetaDataset, GenericMeta): pass


class Dataset(TorchDataset, _DatasetBase, metaclass=MetaDatasetWrapper):
    r"""`torch.utils.data.Dataset` **with extended capabilities.**

    This class inherits from
    `torch.utils.data.Dataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`__,
    co can be used in the same manner after inheritance.
    It allows user to perform the following operations:

    - `cache` - cache all/part of data in memory or on disk
    - `map` - apply function to each element of dataset
    - `apply` - apply function to **all** elements of dataset
    - `reduce` - reduce dataset to single value with specified function

    **Important:**

    - Last cache which is able to hold sample is used. Does not matter whether it's in-memory or on-disk or user-specified.

    - Although multiple cache calls in different parts of `map` should work, users are encouraged to use it as rare as possible and possibly as late as possible for best performance.

    **Example**::

        import torchvision
        from PIL import Image

        # Image loading dataset (use Files for more serious business)
        class Dataset(torchdatasets.Dataset):
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

    def cache(self, cacher: typing.Callable = None):
        r"""**Cache data in memory, disk or specify custom caching.**

        By default all samples are cached in memory. To change this behaviour specify `cacher`
        argument. Some `cacher` implementations can be found in `torchdatasets.cacher` module or you can
        provide your own by inheriting from `torchdatasets.cacher.Cacher` and implementing
        appropriate methods.

        Parameters
        ----------
        cacher : torchdatasets.cacher.Cacher, optional
                Instance of `torchdatasets.cacher.Cacher` (or any other object with compatible interface).
                Check `cacher` module documentation for more information.
                Default: `torchdatasets.cacher.Memory` which caches data in-memory

        Returns
        -------
        Dataset
                Returns self

        """
        if cacher is None:
            cacher = Memory()
        self._cachers.append(cacher)
        self._which.append(len(self._maps))
        return self

    def reduce(self, function: typing.Callable, initializer=None):
        r"""**Reduce dataset to single element with function.**

        Works like `functools.reduce <https://docs.python.org/3/library/functools.html#functools.reduce>`__.

        **Example**::

            class Dataset(torchdatasets.Dataset):
                def __init__(self, max: int):
                    super().__init__() # This is necessary
                    self.range = list(range(max))

                def __getitem__(self, index):
                    return self.range[index]

                def __len__(self):
                    return len(self.range)

            summed_dataset = Dataset(10).reduce(lambda x, y: x + y) # Returns 45


        Parameters
        ----------
        function : typing.Callable
                Two argument function returning single value used to `reduce` dataset.
        initializer: typing.Any, optional
                Value with which reduction will start.

        Returns
        -------
        typing.Any
                Reduced value

        """
        if initializer is None:
            return functools.reduce(function, (item for item in self))
        return functools.reduce(function, (item for item in self), initializer)

    def reset(self, cache: bool = True, maps: bool = True):
        r"""**Reset dataset state.**

        `cache` and `maps` can be resetted separately.

        Parameters
        ----------
        cache : bool, optional
                Reset current cache. Default: `True`
        maps : bool, optional
                Reset current disk cache. Default: `True`

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
    r"""**Concrete** `torchdatasets.Dataset` **responsible for sample-wise concatenation.**

    This class is returned when `|` (logical or operator) is used on instance
    of `torchdatasets.Dataset` (original `torch.utils.data.Dataset
    <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`__ can be used as well).

    **Important:** This class is meant to be more of a proxy for `|` operator,
    you can use it directly though.

    **Example**::

        dataset = (
            torchdatasets.ConcatDataset([dataset1, dataset2, dataset3])
            .map(lambda sample: sample[0] + sample[1] + sample[2]))
        )

    Any `Dataset` methods can be used normally.

    Attributes
    ----------
    datasets : List[Union[torchdatasets.Dataset, torch.utils.data.Dataset]]
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
    r"""**Concrete** `Iterable` **responsible for sample-wise concatenation.**

    This class is returned when `|` (logical or operator) is used on instance
    of `Iterable` (original `torch.utils.data.IterableDataset
    <https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset>`__ can be used as well).

    .. note::

        This class is meant to be more of a proxy for `|` operator,
        you can use it directly though.

    **Example**::

        dataset = (
            torchdatasets.ConcatIterable([dataset1, dataset2, dataset3])
            .map(lambda x, y, z: (x + y, z))
        )

    Any `IterableDataset` methods can be used normally.

    Attributes
    ----------
    datasets : List[Union[torchdatasets.Iterable, torch.utils.data.IterableDataset]]
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
    r"""**Concrete** `torchdatasets.Dataset` **responsible for chaining multiple datasets.**

    This class is returned when `+` (logical or operator) is used on instance
    of `torchdatasets.Dataset` (original `torch.utils.data.Dataset` can be used as well).
    Acts just like PyTorch's `+` or rather `torch.utils.data.ConcatDataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.ConcatDataset>`__

    .. note::

        This class is meant to be more of a proxy for `+` operator,
        you can use it directly though.

    **Example**::

        # Iterate over 3 datasets consecutively
        dataset = torchdatasets.ChainDataset([dataset1, dataset2, dataset3])

    Any `Dataset` methods can be used normally.

    Attributes
    ----------
    datasets : List[Union[torchdatasets.Dataset, torch.utils.data.Dataset]]
            List of datasets to be chained.

    """

    def __init__(self, datasets):
        Dataset.__init__(self)
        TorchConcatDataset.__init__(self, datasets)


class ChainIterable(TorchChain, Iterable):
    r"""**Concrete** `torchdatasets.Iterable` **responsible for chaining multiple datasets.**

    This class is returned when `+` (logical or operator) is used on instance
    of `torchdatasets.Iterable` (original `torch.utils.data.Iterable` can be used as well).
    Acts just like PyTorch's `+` and `ChainDataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.ChainDataset>`__.

    .. note::

        This class is meant to be more of a proxy for `+` operator,
        you can use it directly though.


    **Example**::

        # Iterate over 3 iterable datasets consecutively
        dataset = torchdatasets.ChainDataset([dataset1, dataset2, dataset3])

    Any `Iterable` methods can be used normally.

    Attributes
    ----------
    datasets : List[Union[torchdatasets.Iterable, torch.utils.data.IterableDataset]]
            List of datasets to be chained.

    """

    def __init__(self, datasets):
        Iterable.__init__(self)
        TorchChain.__init__(self, datasets)


###############################################################################
#
#                               CONCRETE CLASSES
#
###############################################################################


class Files(Dataset):
    r"""**Create** `Dataset` **from list of files.**

    Each file is a separate sample. User can use this class directly
    as all necessary methods are implemented.

    `__getitem__` uses Python's `open <https://docs.python.org/3/library/functions.html#open>`__
    and returns file. It's implementation looks like::

        # You can modify open behaviour by passing args nad kwargs to __init__
        with open(self.files[index], *self.args, **self.kwargs) as file:
            return file

    you can use `map` method in order to modify returned `file` or you can overload
    `__getitem__` (image opening example below)::

        import torchdatasets
        import torchvision

        from PIL import Image


        # Image loading dataset
        class ImageDataset(torchdatasets.datasets.FilesDataset):
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
            Arguments saved for `__getitem__`
    **kwargs
            Keyword arguments saved for `__getitem__`

    """

    @classmethod
    def from_folder(cls, path: pathlib.Path, regex: str = "*", *args, **kwargs):
        r"""**Create dataset from** `pathlib.Path` **-like object.**

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
        self.files = files
        self.args = args
        self.kwargs = kwargs

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with open(self.files[index], *self.args, **self.kwargs) as file:
            return file

    def filter(self, predicate: typing.Callable):
        r"""**Remove** `files` **for which predicate returns** `False`**.**

        **Note:** This is different from `torchdatasets.Iterable`'s `filter` method,
        as the filtering is done when called, not during iteration.

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


class TensorDataset(TorchTensorDataset, Dataset):
    r"""**Dataset wrapping** `torch.tensors` **.**

    `cache`, `map` etc. enabled version of `torch.utils.data.TensorDataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.TensorDataset>`__.

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
    expression: Generator expression
            Generator from which one can `yield` via `yield from` syntax.
    """

    def __init__(self, expression):
        super().__init__()
        self.expression = expression

    def __iter__(self):
        yield from self.expression


class _Wrap:
    def __getattr__(self, name):
        return getattr(self.dataset, name)


class WrapDataset(_Wrap, Dataset):
    r"""**Dataset wrapping standard** `torch.data.utils.Dataset` **and making it** `torchdatasets.Dataset` **compatible.**

    All attributes of wrapped dataset can be used normally, for example::

        dataset = td.datasets.WrapDataset(
            torchvision.datasets.MNIST("./data")
        )
        dataset.train # True, has all MNIST attributes

    Parameters:
    -----------
    dataset: `torch.data.utils.Dataset`
            Dataset to be wrapped
    """

    def __init__(self, dataset):
        self.dataset = dataset
        Dataset.__init__(self)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class WrapIterable(_Wrap, Iterable):
    r"""**Iterable wrapping standard** `torch.data.utils.IterableDataset` **and making it** `torchdatasets.Iterable` **compatible.**

    All attributes of wrapped dataset can be used normally as is the case for
    `torchdatasets.datasets.WrapDataset`.

    Parameters:
    -----------
    dataset: `torch.data.utils.Dataset`
            Dataset to be wrapped
    """

    def __init__(self, dataset):
        Iterable.__init__(self)
        self.dataset = dataset

    def __iter__(self):
        yield from self.dataset
