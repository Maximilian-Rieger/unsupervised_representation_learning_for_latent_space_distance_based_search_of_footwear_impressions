import os.path
from PIL import Image   # reading image
import dlib             # face detection
import numpy as np
import pandas as pd     #csv files
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, csv_file, image_dir, transform, type=float):
        """
        Args:
            csv_file (string): Path to the csv file with labels/scores.
            image_dir (string): Directory with all the data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images = pd.read_csv(csv_file, sep=' ', decimal='.')

        self.image_dir = image_dir
        self.transform = transform
        self.type = type

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir,
                                self.images.iloc[idx, 0])

        score = self.type(self.images.iloc[idx, 1])

        # load image
        image = Image.open(img_name)

        # apply transform
        if self.transform:
            image = self.transform(image)

        return image, score