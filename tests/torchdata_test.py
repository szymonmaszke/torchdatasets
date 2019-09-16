import torchdata


class IterableDataset(torchdata.Iterable):
    def __init__(self, start: int, end: int):
        super().__init__()
        self.start: int = start
        self.end: int = end

    def __iter__(self):
        return iter(range(self.start, self.end))


class Dataset(torchdata.Dataset):
    def __init__(self, start: int, end: int):
        super().__init__()
        self.values = list(range(start, end))

    def __getitem__(self, index):
        return self.values[index]

    def __len__(self):
        return len(self.values)


def test_iterable_or():
    # range(1,25) originally, mapped to range(13, 37)
    dataset = IterableDataset(0, 25).map(lambda value: value + 12)
    # Sample-wise concatenation, yields range(13, 37) and range(1, 25)
    for index, (first, second) in enumerate(dataset | IterableDataset(0, 25)):
        assert index + 12 == first
        assert index == second


def test_dataset_add():
    # range(1,25) originally, mapped to range(13, 37)
    dataset = (
        (Dataset(0, 25) | Dataset(0, 25))
        .map(lambda sample: sample[0] * sample[1])
        .cache()
    )
    # Sample-wise concatenation, yields range(13, 37) and range(1, 25)
    for index, value in enumerate(dataset):
        assert index ** 2 == value


def test_dataset_cache():
    # range(1,25) originally, mapped to range(13, 37)
    dataset = (
        Dataset(0, 25)
        .cache()
        .map(lambda sample: sample + sample)
        .cache()
        .map(lambda sample: sample + sample)
        .cache()
    )
    # Sample-wise concatenation, yields range(13, 37) and range(1, 25)
    for _ in range(3):
        for _ in dataset:
            pass

    for index, value in enumerate(dataset):
        smth = index + index
        assert smth + smth == value


def test_dataset_complicated_cache():
    # range(1,25) originally, mapped to range(13, 37)
    dataset = (
        (
            (Dataset(0, 25) | Dataset(0, 25).map(lambda value: value * -1))
            .cache()
            .map(lambda sample: sample[0] + sample[1] + sample[0])
            .cache()
            .map(lambda sample: sample + sample)
            | Dataset(0, 25)
        )
        .cache()
        .map(lambda values: ((values, values), values))
        .map(torchdata.maps.Flatten())
        .cache()
        .map(lambda values: values[1])
        .map(lambda value: value ** 2)
    )
    # Sample-wise concatenation, yields range(13, 37) and range(1, 25)
    for _ in range(3):
        for _ in dataset:
            pass

    for index, value in enumerate(dataset):
        assert index ** 2 == value


class CountingDataset(torchdata.Dataset):
    def __init__(self, max: int):
        super().__init__()  # This is necessary
        self.range = list(range(max))

    def __getitem__(self, index):
        return self.range[index]

    def __len__(self):
        return len(self.range)


def summation(generator):
    return sum(value for value in generator)


def test_apply():
    assert CountingDataset(101).apply(summation) == 5050  # Returns 5050
