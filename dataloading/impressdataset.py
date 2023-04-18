from torch.utils.data import Dataset
import logging

from dataloading.imagedataset import ImageDataset


class ImpressDataset(Dataset):
    """Image dataset."""

    def __init__(self, sole_base_path, shoe_base_path, transforms, limit=None, log_image_info=False):
        """
        Args:
            base_path   (string): Path to the dataset root.
            limit   (int): Optional limit data set size.
        """

        self.shoe_dataset = ImageDataset(shoe_base_path, transform=transforms[0], limit=limit, log_image_info=False)
        self.sole_dataset = ImageDataset(sole_base_path, transform=transforms[1], limit=limit, log_image_info=False,
                                         left="*_3_L.jpg", right="*_1_R.jpg")
        # for sole in self.sole_dataset.data:

        # search files
        logging.info('Found {} sole data'.format(len(self.sole_dataset)))
        logging.info('Found {} shoe data'.format(len(self.shoe_dataset)))

        logging.info('Found {} data'.format(min(len(self.sole_dataset), len(self.shoe_dataset))))

    def __len__(self):
        return min(len(self.sole_dataset), len(self.shoe_dataset))

    def __getitem__(self, idx):
        sole = self.sole_dataset[idx]
        shoe = self.shoe_dataset[idx]

        return sole, shoe
