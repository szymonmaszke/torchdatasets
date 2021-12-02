r"""**This module provides functions one can use with** `torchdatasets.Dataset.map` **method.**

Following `dataset` object will be used throughout documentation for brevity (if not defined explicitly)::

    # Image loading dataset
    import torchdatasets as td

    class Example(td.Dataset):
        def __init__(self, max: int):
            self.values = list(range(max))

        def __getitem__(self, index):
            return self.values[index]

        def __len__(self):
            return len(self.values)

    dataset = Example(100)


`maps` below are general and can be used in various scenarios.

If users want to create their own `map` objects, they can use single argument
callables taking sample and returning sample after modifications.

"""

import typing

from ._base import Base


class After(Base):
    r"""**Apply function after specified number of samples passed.**

    Useful for introducing data augmentation after an initial warm-up period.
    If you want a direct control over when function will be applied to sample,
    please use `torchdatasets.transforms.OnSignal`.

    Example::

        # After 10 samples apply lambda mapping
        dataset = dataset.map(After(10, lambda x: -x))

    Parameters
    ----------
    samples : int
            After how many samples function will start being applied.
    function : Callable
            Function to apply to sample.

    Returns
    -------
    Union[sample, function(sample)]
            Either unchanged sample or function(sample)

    """

    def __init__(self, samples: int, function: typing.Callable):
        self.samples = samples
        self.function = function
        self._elements_counter = -1

    def __call__(self, sample):
        self._elements_counter += 1
        if self._elements_counter > self.samples:
            return self.function(sample)
        return sample


class OnSignal(Base):
    r"""**Apply function based on boolean output of signalling function.**

    Useful for introducing data augmentation after an initial warm-up period.
    You can use it to turn on/off specific augmentation with respect to outer world,
    for example turning on image rotations after 5 epochs and turning off 5 epochs
    before the end in order to fine-tune your network.


    Example::

        import torch
        from PIL import Image

        import torchdatasets as td
        import torchvision


        # Image loading dataset
        class ImageDataset(td.datasets.Files):
            def __getitem__(self, index):
                return Image.open(self.files[index])


        class Handle:
            def __init__(self):
                self.value: bool = False

            def __call__(self):
                return self.value

        # you can change handle.value to switch whether mapping should be applied
        handle = Handle()
        dataset = (
            ImageDataset.from_folder("./data")
            .map(torchvision.transforms.ToTensor())
            .cache()
            # If handle returns True, mapping will be applied
            .map(
                td.maps.OnSignal(
                    handle, lambda image: image + torch.rand_like(image)
                )
            )
        )

    Parameters
    ----------
    signal : Callable
            No argument callable returning boolean, indicating whether to apply function.
    function: Callable
            Function to apply to sample.

    Returns
    -------
    Union[sample, function(sample)]
            Either unchanged sample of function(sample)

    """

    def __init__(self, signal: typing.Callable[..., bool], function: typing.Callable):
        self.signal = signal
        self.function = function

    def __call__(self, sample):
        if self.signal():
            return self.function(sample)
        return sample


class Flatten(Base):
    r"""**Flatten arbitrarily nested sample.**

    Example::

        # Nest elements
        dataset = dataset.map(lambda x: (x, (x, (x, x), x),))
        # Flatten no matter how deep
        dataset = dataset.map(torchdatasets.maps.Flatten())

    Parameters
    ----------
    types : Tuple[type], optional
            Types to be considered non-flat. Those will be recursively flattened.
            Default: `(list, tuple)`

    Returns
    -------
    Tuple[samples]
            Tuple with elements flattened

    """

    def __init__(self, types: typing.Tuple = (list, tuple)):
        self.types = types

    def __call__(self, sample):
        if not isinstance(sample, self.types):
            return sample
        return Flatten._flatten(sample, self.types)

    @staticmethod
    def _flatten(items, types):
        if isinstance(items, tuple):
            items = list(items)

        for index, x in enumerate(items):
            while index < len(items) and isinstance(items[index], types):
                items[index : index + 1] = items[index]
        return tuple(items)


class Repeat(Base):
    r"""**Apply function repeatedly to the sample.**

    Example::

        import torchdatasets as td

        # Creating td.Dataset instance
        ...
        # Increase each value by 10 * 1
        dataset = dataset.map(td.maps.Repeat(10, lambda x: x+1))

    Parameters
    ----------
    n : int
            How many times the function will be applied.
    function : Callable
            Function to apply.

    Returns
    -------
    function(sample)
            Function(sample) applied n times.

    """

    def __init__(self, n: int, function: typing.Callable):
        self.n = n
        self.function = function

    def __call__(self, sample):
        for _ in range(self.n):
            sample = self.function(sample)
        return sample


class _Choice(Base):
    def __init__(self, *indices):
        self.indices = set(indices)

    def _magic_unpack(self, iterable):
        if len(iterable) == 1:
            return iterable[0]
        if len(iterable) == 0:
            return None
        return iterable


class Select(_Choice):
    r"""**Select elements from sample.**

    Sample has to be indexable object (has `__getitem__` method implemented).

    **Important:**

    - Negative indexing is supported if supported by sample object.
    - This function is **faster** than `Drop` and should be used if possible.
    - If you want to select sample from nested `tuple`, please use `Flatten` first
    - Returns single element if only one element is left

    Example::

        # Sample-wise concatenate dataset three times
        new_dataset = dataset | dataset
        # Only second (first index) element will be taken
        selected = new_dataset.map(td.maps.Select(1))

    Parameters
    ----------
    *indices : int
            Indices of objects to select from the sample. If left empty, empty tuple will be returned.

    Returns
    -------
    Tuple[samples]
            Tuple with selected elements

    """

    def __call__(self, sample):
        return self._magic_unpack(tuple(sample[i] for i in self.indices))


class Drop(_Choice):
    r"""**Return sample without selected elements.**

    Sample has to be indexable object (has `__getitem__` method implemented).

    **Important:**

    - Negative indexing is supported if supported by sample object.
    - This function is **slower** than `Select` and the latter should be preffered.
    - If you want to select sample from nested `tuple`, please use `Flatten` first
    - Returns single element if only one element is left
    - Returns `None` if all elements are dropped

    Example::

        # Sample-wise concatenate dataset three times
        new_dataset = dataset | dataset | dataset
        # Zeroth and last samples dropped
        selected = new_dataset.map(td.maps.Drop(0, 2))

    Parameters
    ----------
    *indices : int
            Indices of objects to remove from the sample. If left empty, tuple containing
            all elements will be returned.

    Returns
    -------
    Tuple[samples]
            Tuple without selected elements

    """

    def __call__(self, sample):
        return self._magic_unpack(
            tuple(
                sample[index]
                for index, _ in enumerate(sample)
                if index not in self.indices
            )
        )


class ToAll(Base):
    r"""**Apply function to each element of sample.**

    Sample has to be `iterable` object.

    **Important:**

    If you want to apply function to all nested elements (e.g. in nested `tuple`),
    please use `torchdatasets.maps.Flatten` object first.

    Example::

        # Sample-wise concatenate dataset three times
        new_dataset = dataset | dataset | dataset
        # Each concatenated sample will be increased by 1
        selected = new_dataset.map(td.maps.ToAll(lambda x: x+1))

    Attributes
    ----------
    function : Callable
            Function to apply to each element of sample.

    Returns
    -------
    Tuple[function(subsample)]
            Tuple consisting of subsamples with function applied.

    """

    def __init__(self, function: typing.Callable):
        self.function = function

    def __call__(self, sample):
        return tuple(self.function(subsample) for subsample in sample)


class To(Base):
    """**Apply function to specified elements of sample.**

    Sample has to be `iterable` object.

    **Important:**

    If you want to apply function to all nested elements (e.g. in nested `tuple`),
    please use `torchdatasets.maps.Flatten` object first.

    Example::

        # Sample-wise concatenate dataset three times
        new_dataset = dataset | dataset | dataset
        # Zero and first subsamples will be increased by one, last one left untouched
        selected = new_dataset.map(td.maps.To(lambda x: x+1, 0, 1))

    Attributes
    ----------
    function : Callable
            Function to apply to specified elements of sample.

    *indices : int
            Indices to which function will be applied. If left empty,
            function will not be applied to anything.

    Returns
    -------
    Tuple[function(subsample)]
            Tuple consisting of subsamples with some having the function applied.

    """

    def __init__(self, function: typing.Callable, *indices):
        self.function = function
        self.indices = set(indices)

    def __call__(self, sample):
        return tuple(
            self.function(subsample) if index in self.indices else subsample
            for index, subsample in enumerate(sample)
        )


class Except(Base):
    r"""**Apply function to all elements of sample except the ones specified.**

    Sample has to be `iterable` object.

    **Important:**

    If you want to apply function to all nested elements (e.g. in nested `tuple`),
    please use `torchdatasets.maps.Flatten` object first.

    Example::

        # Sample-wise concatenate dataset three times
        dataset |= dataset
        # Every element increased by one except the first one
        selected = new_dataset.map(td.maps.Except(lambda x: x+1, 0))

    Attributes
    ----------
    function: Callable
            Function to apply to chosen elements of sample.

    *indices: int
            Indices of objects to which function will not be applied. If left empty,
            function will be applied to every element of sample.

    Returns
    -------
    Tuple[function(subsample)]
            Tuple with subsamples where some have the function applied.

    """

    def __init__(self, function: typing.Callable, *indices):
        self.function = function
        self.indices = set(indices)

    def __call__(self, sample):
        return tuple(
            self.function(subsample) if index not in self.indices else subsample
            for index, subsample in enumerate(sample)
        )
