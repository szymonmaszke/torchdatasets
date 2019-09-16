:github_url: https://github.com/szymonmaszke/torchdata

*********
torchdata
*********

**torchdata** is PyTorch oriented library focused on data processing and input pipelines in general.

It extends `torch.utils.data.Dataset` and equips it with
functionalities known from `tensorflow.data <https://www.tensorflow.org/api_docs/python/tf/data/Dataset>`__
like `map` or `cache`.
All of that with minimal interference (single call to `super().__init__()`) with original
PyTorch's datasets.

Some functionalities:

* `map` or `apply` arbitrary functions to dataset
* `cache` allows you to cache data in memory or on disk (even partially, say first `20%`)
* Full `torch.utils.data.IterableDataset` and `torch.utils.data.Dataset` support
* Easy to create custom methods of caching, choosing elements to cache, maps and datasets
* Concrete and base classes designed for file reading and other general tasks

If you are looking for ecosystem of supporting
functions around PyTorch check `torchfunc <https://github.com/szymonmaszke/torchfunc>`__.

Modules
#######

.. toctree::
   :glob:
   :maxdepth: 1

   packages/*
   
.. toctree::
   :hidden:

   related


Installation
############

Following installation methods are available:

`pip: <https://pypi.org/project/torchdata/>`__
==============================================

To install latest release:

.. code-block:: shell

  pip install --user torchdata

To install `nightly` version:

.. code-block:: shell

  pip install --user torchdata-nightly


`Docker: <https://cloud.docker.com/repository/docker/szymonmaszke/torchdata>`__
===============================================================================

Various `torchdata` images are available both CPU and GPU-enabled.
You can find them in Docker Cloud at `szymonmaszke/torchdata`

CPU
---

CPU image is based on `ubuntu:18.04 <https://hub.docker.com/_/ubuntu>`__ and
official release can be pulled with:

.. code-block:: shell
  
  docker pull szymonmaszke/torchdata:18.04

For `nightly` release:

.. code-block:: shell
  
  docker pull szymonmaszke/torchdata:nightly_18.04

This image is significantly lighter due to lack of GPU support.

GPU
---

All images are based on `nvidia/cuda <https://hub.docker.com/r/nvidia/cuda/>`__ Docker image.
Each has corresponding CUDA version tag ( `10.1`, `10` and `9.2`) CUDNN7 support
and base image ( `ubuntu:18.04 <https://hub.docker.com/_/ubuntu>`__ ).

Following images are available:

- `10.1-cudnn7-runtime-ubuntu18.04`
- `10.1-runtime-ubuntu18.04`
- `10.0-cudnn7-runtime-ubuntu18.04`
- `10.0-runtime-ubuntu18.04`
- `9.2-cudnn7-runtime-ubuntu18.04`
- `9.2-runtime-ubuntu18.04`

Example pull:

.. code-block:: shell
  
  docker pull szymonmaszke/torchdata:10.1-cudnn7-runtime-ubuntu18.04

You can use `nightly` builds as well, just prefix the tag with `nightly_`, for example
like this:

.. code-block:: shell
  
  docker pull szymonmaszke/torchdata:nightly_10.1-cudnn7-runtime-ubuntu18.04


`conda: <https://anaconda.org/conda-forge/torchdata>`__
=======================================================

**TO BE ADDED**

.. code-block:: shell

  conda install -c conda-forge torchdata
