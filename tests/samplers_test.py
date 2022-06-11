import itertools

import pytest
import torch
import torchdatasets


def create_inputs(high: int):
    return torch.randint(high=high, size=(100,))


def verify_example(sampler, inputs):
    count = torch.bincount(torch.tensor([inputs[index] for index in sampler]))
    return torch.all(count == torch.max(count))


@pytest.mark.parametrize(
    "sampler_cls,inputs",
    list(
        itertools.product(
            (
                torchdatasets.samplers.RandomOverSampler,
                torchdatasets.samplers.RandomUnderSampler,
            ),
            [create_inputs(i) for i in [1, 2, 5, 7]],
        )
    ),
)
def test_samplers(sampler_cls, inputs):
    sampler = sampler_cls(inputs)
    assert verify_example(sampler, inputs)


@pytest.mark.parametrize(
    "inputs",
    [create_inputs(i) for i in [1, 2, 5, 7]],
)
def test_weighted_imbalanced_sampler(inputs):
    sampler = torchdatasets.samplers.WeightedImbalancedSampler(inputs, num_samples=50)
    for _ in sampler:
        pass
