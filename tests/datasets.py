import torchdata


class ExampleIterable(torchdata.Iterable):
    def __init__(self, start: int, end: int):
        super().__init__()
        self.start: int = start
        self.end: int = end

    def __iter__(self):
        return iter(range(self.start, self.end))


class ExampleDataset(torchdata.Dataset):
    def __init__(self, start: int, end: int):
        super().__init__()
        self.values = list(range(start, end))

    def __getitem__(self, index):
        return self.values[index]

    def __len__(self):
        return len(self.values)
