from torch.utils.data import Dataset, Subset
import random 

def filter_dataset(dataset, filter_fn):
    indices = [i for i, x in enumerate(dataset) if filter_fn(x)]
    return Subset(dataset, indices)

def random_subset(dataset, n):
    indices = random.sample(range(len(dataset)), n)
    return Subset(dataset, indices)
class RepeatDataset(Dataset):
    def __init__(self, base_dataset, length):
        self.base = base_dataset
        self.length = length

    def __getitem__(self, index):
        if index >= self.length:
            raise IndexError("Index out of bounds")

        return self.base[index % len(self.base)]

    def __len__(self):
        return self.length 

class TransformDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.base = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.base[index], index)

    def __len__(self):
        return len(self.base)