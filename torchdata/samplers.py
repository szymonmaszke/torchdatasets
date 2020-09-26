r"""**This module implements samplers to be used in conjunction with** `torch.utils.data.DataLoader` **instances**.

Those can be used just like PyTorch's `torch.utils.data.Sampler` instances.

See `PyTorch tutorial <https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler>`__
for more examples and information.

"""

import builtins

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
    indices : typing.Iterable
            A sequence of indices
    replacement : bool, optional
            Samples are drawn with replacement if `True`. Default: `False`
    num_samples : int, optional
            Number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is `True`.
            Default: `None`
    """

    def __init__(self, indices, replacement=False, num_samples=None):
        RandomSampler.__init__(self, indices, replacement, num_samples)

    def __iter__(self):
        for index in RandomSampler.__iter__(self):
            yield self.data_source[index]


class _Equalizer(Sampler):
    def __init__(self, labels: torch.tensor, function):
        if len(labels.shape) > 1:
            raise ValueError(
                "labels can only have a single dimension (N, ), got shape: {}".format(
                    labels.shape
                )
            )
        tensors = [
            torch.nonzero(labels == i, as_tuple=False).flatten()
            for i in torch.unique(labels)
        ]
        self.samples_per_label = getattr(builtins, function)(map(len, tensors))
        self.samplers = [
            iter(
                RandomSubsetSampler(
                    tensor,
                    replacement=len(tensor) < self.samples_per_label,
                    num_samples=self.samples_per_label
                    if len(tensor) < self.samples_per_label
                    else None,
                )
            )
            for tensor in tensors
        ]

    @property
    def num_samples(self):
        return self.samples_per_label * len(self.samplers)

    def __iter__(self):
        for _ in range(self.samples_per_label):
            for index in torch.randperm(len(self.samplers)).tolist():
                yield next(self.samplers[index])

    def __len__(self):
        return self.num_samples


class WeightedImbalancedSampler(torch.utils.data.WeightedRandomSampler):
    r"""**Sample elements using per-class weights.**

    Data points with underrepresented classes will have higher probability
    of being chosen.

    .. note::

        Labels (possibly multiclass) have to be of shape `(N,)`
        (single dimension). No additional dimensions allowed.

    Parameters
    ----------
    labels : torch.Tensor
            Tensor containing labels for respective samples.
    """

    def __init__(self, labels, num_samples: int):
        super().__init__(
            weights=(
                torch.nn.functional.one_hot(labels)
                * (1 / torch.bincount(labels).float())
            ).sum(dim=1),
            num_samples=num_samples,
        )


class RandomOverSampler(_Equalizer):
    r"""**Sample elements randomly with underrepresented classes upsampled.**

    Length is equal to `max_samples_per_class * classes`.

    .. note::

        Labels (possibly multiclass) have to be of shape `(N,)`
        (single dimension). No additional dimensions allowed.


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

    .. note::

        Labels (possibly multiclass) have to be of shape `(N,)`
        (single dimension). No additional dimensions allowed.

    Parameters
    ----------
    labels : torch.Tensor
            Tensor containing labels for respective samples.
    """

    def __init__(self, labels: torch.tensor):
        super().__init__(labels, "min")


class Distribution(Sampler):
    r"""**Sample** `num_samples` **indices from distribution object.**

    Parameters
    ----------
    distribution : torch.distributions.distribution.Distribution
            Distribution-like object implementing `sample()` method.
    num_samples : int
            Number of samples to be yielded

    """

    def __init__(
        self,
        distribution: torch.distributions.distribution.Distribution,
        num_samples: int,
    ):
        self.distribution = distribution
        self.num_samples = num_samples

    def __iter__(self):
        for _ in range(self.num_samples):
            yield self.distribution.sample()

    def __len__(self):
        return self.num_samples
