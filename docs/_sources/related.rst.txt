****************
Related projects
****************

Below you can find other projects started by the same author and based on `PyTorch <https://pytorch.org/>`__ as well:

`torchlayers: <https://github.com/szymonmaszke/torchlayers>`__
==============================================================

**torchlayers** is a library based on PyTorch
providing **automatic shape and dimensionality inference of** `torch.nn` **layers** + additional
building blocks featured in current SOTA architectures (e.g. `Efficient-Net <https://arxiv.org/abs/1905.11946>`__).

* Shape inference for most of `torch.nn` module (convolutional, recurrent, transformer, attention and linear layers)
* Dimensionality inference (e.g. `torchlayers.Conv` working as `torch.nn.Conv1d/2d/3d` based on `input shape`)
* Shape inference of user created modules
* Additional `Keras-like <https://www.tensorflow.org/guide/keras>`__ layers (e.g. `torchlayers.Reshape` or `torchlayers.StandardNormalNoise`)
* Additional SOTA layers mostly from ImageNet competitions (e.g. `PolyNet <https://arxiv.org/abs/1608.06993>`__, `Squeeze-And-Excitation <https://arxiv.org/abs/1709.01507>`__, `StochasticDepth <www.arxiv.org/abs/1512.03385>`__.
* Useful defaults (`same` padding and default `kernel_size=3` for `Conv`, dropout rates etc.)
* Zero overhead and `torchscript <https://pytorch.org/docs/stable/jit.html>`__ support

You can read documentation over at https://github.com/szymonmaszke/torchlayers.

`torchlambda: <https://github.com/szymonmaszke/torchlambda>`__
==============================================================

**torchlambda** is a tool to deploy PyTorch models on Amazon's AWS Lambda using AWS SDK for C++ and custom C++ runtime.

* Using statically compiled dependencies whole package is shrunk to only 30MB.
* Due to small size of compiled source code users can pass their models as AWS Lambda layers. Services like Amazon S3 are no longer necessary to load your model.
* torchlambda has it's PyTorch & AWS dependencies always up to date because of continuous deployment run at 03:00 a.m. every day.

You can read project's wiki over at https://github.com/szymonmaszke/torchlambda/wiki

`torchfunc: <https://github.com/szymonmaszke/torchfunc>`__
==========================================================

**torchfunc** is a PyTorch oriented library with a goal to help you with:

* Improve and analyse performance of your neural network
* Record/analyse internal state of torch.nn.Module as data passes through it
* Do the above based on external conditions (using single Callable to specify it)
* Day-to-day neural network related duties (model size, seeding, time measurements etc.)
* Get information about your host operating system, torch.nn.Module device, CUDA capabilities etc.

It **is not** directly related to model creation but should be considered more of an environment
around this process.

You can read documentation over at https://szymonmaszke.github.io/torchfunc.
