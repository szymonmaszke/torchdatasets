import torch

import torchdata


def create_example(over: bool):
    labels = torch.tensor([0, 0, 0, 1, 0, 0, 1])
    sampler_class = getattr(
        torchdata.samplers, "Random" + ("Over" if over else "Under") + "Sampler"
    )
    sampler = sampler_class(labels)
    return torch.tensor([labels[index] for index in sampler])


def test_random_oversampler():
    oversampled = create_example(True)
    assert len(oversampled) == 2 * 5
    assert (oversampled == 0).sum() == (oversampled == 1).sum()


def test_random_undersampler():
    undersampled = create_example(False)
    assert len(undersampled) == 2 * 2
    assert (undersampled == 0).sum() == (undersampled == 1).sum()
