import torch
import torchdatasets


class ExampleIterable(torchdatasets.Iterable):
    def __init__(self, start: int, end: int):
        super().__init__()
        self.start: int = start
        self.end: int = end

    def __iter__(self):
        return iter(range(self.start, self.end))


class ExampleDataset(torchdatasets.Dataset):
    def __init__(self, start: int, end: int):
        super().__init__()
        self.values = list(range(start, end))

    def __getitem__(self, index):
        return self.values[index]

    def __len__(self):
        return len(self.values)


class ExampleTensorDataset(torchdatasets.Dataset):
    def __init__(self, size):
        super().__init__()
        self.values = torch.randn(size, 5)

    def __getitem__(self, index):
        return self.values[index]

    def __len__(self):
        return len(self.values)
