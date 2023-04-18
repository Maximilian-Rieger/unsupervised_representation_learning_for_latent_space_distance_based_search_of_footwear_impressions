import torch.utils.data as data
import numpy as np


class NpzDataset(data.Dataset):

    def __init__(self, file_path, key=None, transform=None):
        with np.load(file_path) as data:
            self.data = data[key] if key else data

        self.transform = transform

    def __getitem__(self, index):

        elem = self.data[index]
        if self.transform is not None:
            elem = self.transform(elem)

        return elem

    def __len__(self):
        return len(self.data)