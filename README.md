![torchdata Logo](https://github.com/szymonmaszke/torchdata/blob/master/assets/banner.png)

--------------------------------------------------------------------------------

| Version | Docs | Tests | Coverage | Style | PyPI | Python | PyTorch | Docker | Roadmap |
|---------------|-------|----------|-------|------|--------|---------|--------|---------|---------|
| [![Version](https://img.shields.io/static/v1?label=&message=0.1.0&color=377EF0&style=for-the-badge)](https://github.com/szymonmaszke/torchdata/releases) | [![Documentation](https://img.shields.io/static/v1?label=&message=docs&color=EE4C2C&style=for-the-badge)](https://szymonmaszke.github.io/torchdata/)  | ![Tests](https://github.com/szymonmaszke/torchdata/workflows/test/badge.svg) | | | [![PyPI](https://img.shields.io/static/v1?label=&message=PyPI&color=377EF0&style=for-the-badge)](https://pypi.org/project/torchdata/) | [![Python](https://img.shields.io/static/v1?label=&message=3.7&color=377EF0&style=for-the-badge&logo=python&logoColor=F8C63D)](https://www.python.org/) | [![PyTorch](https://img.shields.io/static/v1?label=&message=1.2.0&color=EE4C2C&style=for-the-badge)](https://pytorch.org/) | [![Docker](https://img.shields.io/static/v1?label=&message=docker&color=309cef&style=for-the-badge)](https://cloud.docker.com/u/szymonmaszke/repository/docker/szymonmaszke/torchdata) | [![Roadmap](https://img.shields.io/static/v1?label=&message=roadmap&color=009688&style=for-the-badge)](https://github.com/szymonmaszke/torchdata/blob/master/ROADMAP.md) |

[__torchdata__](https://szymonmaszke.github.io/torchdata/) is [PyTorch](https://pytorch.org/) oriented library focused on data processing and input pipelines in general.

It extends `torch.utils.data.Dataset` and equips it with
functionalities known from [tensorflow.data](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)
like `map` or `cache` (with some additions unavailable in aforementioned) .
All of that with minimal interference (single call to `super().__init__()`) in original
PyTorch's datasets.

### Functionalities overview:

* `map` or `apply` arbitrary functions to dataset
* `cache` allows you to cache data in memory or on disk (even partially, say first `20%`)
* Full `torch.utils.data.IterableDataset` and `torch.utils.data.Dataset` support
* Easy to create custom methods of caching, choosing elements to cache, maps and datasets
* Concrete and base classes designed for file reading and other general tasks

# Quick examples

- Create image dataset, convert it to Tensors, cache and concatenate with smoothed labels:

```python
# Imports assumed
# Example dataset return all 1 labels
class Labels(torchdata.Dataset):
    def __init__(self, length):
        self.length = length
        super().__init__()

    def __getitem__(self, _):
        return 1

    def __len__(self):
        return len(length)


# Convenience class based on torchdata.Dataset
class ImageDataset(torchdata.Files):
    def __getitem__(self, index):
        return Image.open(self.files[index])


images = (
    ImageDataset.from_folder("./data").map(torchvision.transforms.ToTensor()).cache()
)

smoothed_labels = Labels(len(images)).map(lambda label: label - 0.1)

# That's how you concatenate sample-wise
for image, label in images | smoothed_labels:
    pass
```

- Cache first `1000` samples in memory, save the rest on disk in folder `./cache`:

```python
images = (
    ImageDataset.from_folder("./data").map(torchvision.transforms.ToTensor())
    # First 1000 samples in memory
    .cache(torchdata.modifiers.UpToIndex(torchdata.cachers.Memory(), 1000))
    # Sample from 1000 to the end saved with Pickle on disk
    .cache(torchdata.modifiers.FromIndex(torchdata.cachers.Pickle("./cache"), 1000))
    # You can define your own cachers, modifiers, see docs
)
```
To see what else you can do please check [**torchdata documentation**](https://szymonmaszke.github.io/torchdata/)

# Installation

## [pip](<https://pypi.org/project/torchdata/>)

### Latest release:

```shell
pip install --user torchdata
```

### Nightly:

```shell
pip install --user torchdata-nightly
```

## [Docker](https://cloud.docker.com/repository/docker/szymonmaszke/torchdata)

__CPU standalone__ and various versions of __GPU enabled__ images are available
at [dockerhub](https://cloud.docker.com/repository/docker/szymonmaszke/torchdata).

For CPU quickstart, issue:

```shell  
docker pull szymonmaszke/torchdata:18.04
```

Nightly builds are also available, just prefix tag with `nightly_`. If you are going for `GPU` image make sure you have
[nvidia/docker](https://github.com/NVIDIA/nvidia-docker) installed and it's runtime set.

# Contributing

If you find any issue or you think some functionality may be useful to others and fits this library, please [open new Issue](https://help.github.com/en/articles/creating-an-issue) or [create Pull Request](https://help.github.com/en/articles/creating-a-pull-request-from-a-fork).

To get an overview of something which one can done to help this project, see [Roadmap](https://github.com/szymonmaszke/torchdata/blob/master/ROADMAP.md)
