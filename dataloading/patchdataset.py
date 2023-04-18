import os, glob
from PIL import Image
from torch.utils.data import Dataset
import logging


class PatchDataset(Dataset):
    """Image dataset."""

    def __init__(self, base_path, transform, limit=None, log_image_info=None, cache_data=None):
        """
        Args:
            base_path   (string): Path to the dataset root.
            limit   (int): Optional limit data set size.
        """

        self.base_path = base_path
        self.transform = transform
        self.data = []
        patches = os.path.join(base_path, "*")

        # search files
        patches = glob.glob(patches)
        patches.sort()

        if log_image_info is not None:
            logging.info('{} {}'.format(base_path, len(patches)))

        if limit is not None:
            images = patches[:limit]

        self.data = patches
        if log_image_info == 'full':
            image = patches[0]
            image = Image.open(image)
            logging.info('Input image: {}x{} mode: {}'.format(image.height, image.width, image.mode))
            if self.transform:
                image = self.transform(image)
            logging.info('Input image tensor: {}, dtype: {}'.format(image.shape, image.dtype))

    def load_image(self, image):
        image = Image.open(image)

        # apply Transforms
        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            if idx.start <= idx.stop:
                images = self.data[idx]
            else:
                images = self.data[slice(idx.start, len(self.data))] + self.data[slice(0, idx.stop)]
            return [self.load_image(image) for image in images]
        return self.load_image(self.data[idx])
