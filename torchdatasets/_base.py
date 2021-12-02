import abc
import functools
import typing

import torch
from torch.utils.data import ConcatDataset as TorchConcatDataset
from torch.utils.data import Dataset as TorchDataset

from ._dev_utils import apply_mapping, reversed_enumerate

################################################################################
#
#                               METACLASSES
#
################################################################################


class MetaIterable(type):
    """MetaClass allowing objects to perform dataset related operations.

    Operations implemented by MetaIterable
    - map - apply function to each element of dataset

    WARNING: You __should not__ use this metaclass in your code.
    If you wish to use provided functionality, please inherit Iterable class [ref].

    """

    def __new__(cls, name, bases, namespace, *args):
        iterable = super().__new__(cls, name, bases, namespace)
        setattr(
            iterable,
            "__iter__",
            MetaIterable.create__iter__(getattr(iterable, "__iter__")),
        )
        return iterable

    @staticmethod
    def create__iter__(iter_function):
        """Override default __iter__ to enable filtering and mapping."""

        @functools.wraps(iter_function)
        def __iter__(self):
            for sample in iter_function(self):
                for i, filter_function in enumerate(self._filters, 1):
                    sample = apply_mapping(
                        sample, self._maps, self._which[i - 1], self._which[i]
                    )
                    if not filter_function(sample):
                        break
                else:
                    yield apply_mapping(
                        sample, self._maps, self._which[-1], len(self._maps)
                    )

        return __iter__


class MetaDataset(type):
    """MetaClass allowing objects to perform dataset related operations.

    Operations implemented by MetaBase:
    - cache - cache data on disk or RAM or user specified way

    WARNING: You **should not** use this metaclass in your code.
    If you wish to use provided functionality, please inherit Base class [ref].

    """

    def __new__(cls, name, bases, namespace, *args):
        dataset = super().__new__(cls, name, bases, namespace)
        # To be fixed?
        setattr(
            dataset,
            "__getitem__",
            MetaDataset.create__getitem__(getattr(dataset, "__getitem__")),
        )
        return dataset

    @staticmethod
    def create__getitem__(getitem_function):
        """Override default __getitem__ to enable caching and mapping.

        To see implementation check _dev_utils/base.py.
        """

        def get_sample(self, index, original_getitem):
            # Check whether available in cache, going from latest
            for cacher_index, cacher in reversed_enumerate(self._cachers):
                if index in cacher:
                    # If so, return from cache and index in mappings when this cache was made.
                    return cacher[index], self._which[cacher_index]

            # Not available in any cache, get from original function
            sample = original_getitem(self, index)

            # Allows to iterate over mappings only once always
            most_mappings = 0
            for cacher_index, cacher in enumerate(self._cachers):
                # Get how many maps to apply before caching
                maps_index = self._which[cacher_index]
                sample = apply_mapping(
                    sample, self._maps, start=most_mappings, end=maps_index
                )
                most_mappings = maps_index
                # Try to cache with this cacher
                cacher[index] = sample
                # If cached by this cacher, return

            # Return sample with most_mappings applied (possibly all)
            return sample, most_mappings

        @functools.wraps(getitem_function)
        def __getitem__(self, index):
            sample, maps_start = get_sample(self, index, getitem_function)
            return apply_mapping(sample, self._maps, maps_start, len(self._maps))

        return __getitem__


################################################################################
#
#                               SHARED BASE CLASSES
#
################################################################################


class Base:
    def __str__(self) -> str:
        return "{}.{}".format(type(self).__module__, type(self).__name__)

    def __repr__(self) -> str:
        parameters = ", ".join(
            "{}={}".format(key, value)
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        )
        return "{}({})".format(self, parameters)
