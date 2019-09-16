r"""**This module implements samplers to be used in conjunction with** `torch.utils.data.DataLoader` **instances**.

Those can be used just like PyTorch's `torch.utils.data.Sampler` instances.

See https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler
for more information.

"""

import builtins
import dataclasses

import torch
from torch.utils.data import RandomSampler, Sampler, SubsetRandomSampler

from ._base import Base


# Source of mixed class below:
# https://github.com/pytorch/pytorch/blob/master/torch/utils/data/sampler.py#L68
class RandomSubsetSampler(Base, RandomSampler):
    r"""**Sample elements randomly from a given list of indices.**

    If without `replacement`, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Similar to PyTorch's `SubsetRandomSampler`, but this one allows you to specify
    `indices` which will be sampled in random order, not `range` subsampled.

    Parameters
    ----------
    indices : Iterable
            A sequence of indices
    replacement : bool, optional
            Samples are drawn with replacement if ``True``. Default: ``False``
    num_samples : int, optional
            number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
    """

    def __init__(self, indices, replacement=False, num_samples=None):
        RandomSampler.__init__(self, indices, replacement, num_samples)

    def __iter__(self):
        for index in RandomSampler.__iter__(self):
            yield self.data_source[index]


class _Equalizer(Sampler):
    def __init__(self, labels: torch.tensor, function):
        tensors = [torch.nonzero(labels == i).flatten() for i in torch.unique(labels)]
        self.samples_per_label = getattr(builtins, function)(map(len, tensors))
        self.samplers = [
            RandomSubsetSampler(
                tensor, replacement=True, num_samples=self.samples_per_label
            )
            for tensor in tensors
        ]

    @property
    def num_samples(self):
        return torch.cumsum([len(sampler) for sampler in self.samplers])

    def __iter__(self):
        for indices in zip(self.samplers):
            for index in indices:
                yield index

    def __len__(self):
        return self.num_samples


class RandomOverSampler(_Equalizer):
    r"""**Sample elements randomly with underrepresented classes upsampled.**

    Length is equal to `max_samples_per_class * classes`.

    Parameters
    ----------
    labels : torch.Tensor
            Tensor containing labels for respective samples.
    """

    def __init__(self, labels):
        super().__init__(labels, "max")


class RandomUnderSampler(_Equalizer):
    r"""**Sample elements randomly with overrepresnted classes downsampled.**

    Length is equal to `min_samples_per_class * classes`.

    Parameters
    ----------
    labels : torch.Tensor
            Tensor containing labels for respective samples.
    """

    def __init__(self, labels: torch.tensor):
        super().__init__(labels, "min")


@dataclasses.dataclass
class Distribution(Sampler):
    r"""**Sample** `num_samples` **indices from distribution object.**

    Parameters
    ----------
    distribution : torch.distributions.distribution.Distribution
            Distribution-like object implementing `sample()` method.
    num_samples : int
            Number of samples to be yielded

    """

    distribution: torch.distributions.distribution.Distribution
    num_samples: int

    def __iter__(self):
        for _ in range(self.num_samples):
            yield self.distribution.sample()

    def __len__(self):
        return self.num_samples
