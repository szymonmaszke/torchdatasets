# Roadmap

This document highlights possible improvements and additional implementations
driving this library towards next release (version `0.2.0`).

You can check whether someone started work on this task in issues under the label
`roadmap` and specific `improvement` or `implementation`.

Any help with tasks described below will be highly appreciated, thanks!

## Improvements

Following tasks might be considered improvements over what is already done.
Please use label `roadmap` and `improvement` and describe the task you are willing to take
(or describe new task you would consider an improvement).

- Check and test support for multiprocessing and PyTorch's workers
- Highly increase test coverage (`modifiers`, `maps` and `cachers`)
- Fix documentation and make it more readable
- Testing with [Hypothesis](https://github.com/HypothesisWorks/hypothesis)
- Tutorials creation
- Conda release

## Implementations

- General `modifiers`, `cachers` and `maps` if any deemed missing.
- Cloud-oriented `cacher`, which would allow for easier incorporation
of `torchdata` in more challenging tasks (possible as another repository)
- Less I/O intesive disk oriented `cachers` (e.g. save 100 samples as one file)
