from torch.utils.data import Dataset
import logging
import os, glob
from PIL import Image  # reading image
from operator import itemgetter as get

corretcly_transformed = ['1_L', '1_R', '3_L', '3_R', '5_R', '7_L', '7_R', '8_L', '10_L', '10_R', '14_R', '15_R', '17_R', '18_L', '18_R', '19_L', '19_R', '21_L', '21_R', '22_R', '24_R', '25_L', '25_R', '26_L', '26_R', '28_L', '28_R', '29_L', '30_R', '32_R', '34_L', '37_L', '38_R', '39_L', '39_R', '40_L', '40_R', '42_L', '43_L', '43_R', '44_L', '44_R', '47_L', '49_L', '51_L', '51_R', '52_L', '52_R', '53_L', '53_R', '54_L', '54_R', '55_L', '55_R', '57_L', '58_L', '58_R', '59_L', '60_L', '60_R', '61_L', '62_R', '64_L', '64_R', '65_L', '66_L', '66_R', '67_L', '67_R', '68_L', '68_R', '69_L', '70_L']


class ImpressAlignedDataset(Dataset):
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

        data = [e for e in data if any([ct in e['image'] for ct in corretcly_transformed])]
        # data = list(filter(None, data))

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
