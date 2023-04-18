import os, glob
from PIL import Image  # reading image
from torch.utils.data import Dataset
import logging
import re
from functools import lru_cache
from random import shuffle
from dataloading.imagedataset import ImageDataset


class DeAugmentingImpressDataset(Dataset):
    def __init__(self,
                 base_path,
                 transform,
                 sets=None,
                 limit=None,
                 offset=None,
                 log_image_info=False,
                 pattern=None,
                 cache=False,
                 path_transform=None
                 ):
        if sets is None:
            sets = ['prescaled-S', 'clean']
        if pattern is None:
            pattern = {
                'images': ['impress_*_*_*_*.png'],
            }
        if path_transform is None:
            path_transform = lambda path: int(re.match(r'.*impress_(\d{1,3})', path).group(1))

        kwargs = {
            'return_path': True,
            'log_image_info': log_image_info,
            'limit': limit,
            'pattern': pattern,
            'offset': offset,
            'cache': cache,
            'shuffle_data': cache,
            'path_transform': path_transform,
        }
        self.augmented = ImageDataset(base_path, transform, set=sets[0], **kwargs)
        self.original = ImageDataset(base_path, transform, set=sets[1], **kwargs)
        self.origin_cache = {}
        for img_path in self.original.data:
            label = path_transform(img_path)
            self.origin_cache.update({label: img_path})

    def get_origin(self, label):
        original_path = self.origin_cache[label]
        ori_label, original = self.original.load_image(original_path)
        return ori_label, original

    def __len__(self):
        return len(self.augmented.data)

    def __getitem__(self, idx):
        res = self.augmented[idx]
        if isinstance(idx, slice):
            return [(img, self.get_origin(label)[1]) for label, img in res]
        aug_label, augmented = res
        ori_label, original = self.get_origin(aug_label)

        return augmented, original
