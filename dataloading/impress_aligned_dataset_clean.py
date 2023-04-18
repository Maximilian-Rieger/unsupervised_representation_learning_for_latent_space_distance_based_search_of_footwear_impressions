from torch.utils.data import Dataset
import logging
import os, glob
from PIL import Image  # reading image
from operator import itemgetter as get

class ImpressAlignedDatasetClean(Dataset):
    """Image dataset."""

    def __init__(self, base_path, transforms, limit=None, log_image_info=False,
                 images_exp='*_*_merge.shoe.jpg'):
        """
        Args:
            base_path   (string): Path to the dataset root.
            limit   (int): Optional limit data set size.
        """

        self.transform = transforms
        images_path = os.path.join(base_path, "*", images_exp)

        # search files
        if log_image_info:
            logging.info('Found {} image data'.format(len(images_path)))

        images = glob.glob(images_path)
        images.sort()

        data = [{
            'image': image,
            'impression': image.replace('.shoe', '.impression.threshold'),
        } for image in images]

        if limit is not None:
            data = data[:limit]

        self.data = data

    def load_image(self, entry):
        image, impression = get('image', 'impression')(entry)
        image = Image.open(image)
        impression = Image.open(impression)

        # apply Transforms
        if self.transform:
            img_trans, imp_trans = get('image', 'impression')(self.transform)
            image = img_trans(image)
            impression = imp_trans(impression)

        return image, impression

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
