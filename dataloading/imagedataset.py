import os, glob
from PIL import Image  # reading image
from torch.utils.data import Dataset
import logging
import re
from functools import lru_cache
from random import shuffle


class ImageDataset(Dataset):
    """Image dataset."""

    def __init__(
            self, base_path, transform, limit=None, offset=None, log_image_info=False, pattern=None, shared_pattern=None,
            set=None, filter_cb=None, cache=False, return_path=False, shuffle_data=False, path_transform=None, transform_path_count=None):
        """
        Args:
            base_path   (string): Path to the dataset root.
            limit   (int): Optional limit data set size.
            offset  (int): Optional offset data set size.
            log_image_info (bool): Log image information.
            pattern (str): Pattern to search for images.
            shared_pattern (str): Shared pattern to search for images.
            set (str): Set to load.
            filter_cb (function): Optional filter callback.
            cache (bool): Cache loaded images.
            return_path (bool): Return path to image.
            shuffle_data (bool): Shuffle data.
            path_transform (function): Optional path transform.
            transform_path_count (int): Optional number of transformed images to return.
        """

        if pattern is None:
            pattern = {'left': "*_*_2.jpg", 'right': "*_*_3.jpg"}
        if shared_pattern is None:
            shared_pattern = set or '*'

        self.cache = cache
        self.return_path = return_path
        self.path_transform = path_transform
        self.base_path = base_path
        self.transform = transform
        self.data = []
        self.pattern = pattern
        self.shared_pattern = shared_pattern
        self.set = set
        self.filter_cb = filter_cb
        self.shuffle_data = shuffle_data
        self.transform_path_count = transform_path_count or 1

        extended_base_path = os.path.join(base_path, shared_pattern)
        patterns_loaded = ImageDataset.load_patterns(extended_base_path, pattern)

        images = ImageDataset.merge_patterns(patterns_loaded)
        if filter_cb is not None:
            if type(filter_cb) is str:
                images = list(filter(lambda img: not re.search(r"{}".format(filter_cb), img), images))
            elif callable(filter_cb):
                images = list(filter(filter_cb, images))

        if not shuffle_data:
            images.sort()
        else:
            shuffle(images)

        if log_image_info:
            ImageDataset.image_info(patterns_loaded)

        data_len = len(images)
        logging.info('Found {} data'.format(data_len))
        if offset is None:
            offset = 0
        if limit is None:
            limit = data_len
        images = images[offset:limit]
        if offset != 0 or limit != data_len:
            logging.info('Using slice of ({}:{}) data'.format(offset, limit))

        self.data = images
        if log_image_info:
            image = images[0]
            image = Image.open(image)
            logging.info('Input image: {}x{} mode: {}'.format(image.height, image.width, image.mode))
            if self.transform:
                image = self.transform(image)
            logging.info('Input image tensor: {}, dtype: {}'.format(image.shape, image.dtype))

        def internal_load_image_cached(image):
            return self.load_image_cached(image)

        def internal_load_image_with_path_cached(image):
            res = self.load_image_cached(image)
            if self.path_transform is not None:
                image = self.path_transform(image)
            return image, res

        def internal_load_image(image):
            return self.load_image_normal(image)

        def internal_load_image_with_path(image):
            res = self.load_image_normal(image)
            if self.path_transform is not None:
                image = self.path_transform(image)
            return image, res
        
        if cache and return_path:
            self._load_image = internal_load_image_with_path_cached
        elif cache and not return_path:
            self._load_image = internal_load_image_cached
        elif not cache and return_path:
            self._load_image = internal_load_image_with_path
        else:
            self._load_image = internal_load_image

    @staticmethod
    def load_patterns(base_path, patterns):
        """

        :type base_path: str
        :type patterns: union(str, dict)
        """
        images = {}
        if type(patterns) is dict:
            for pattern_name, pattern in patterns.items():
                logging.debug('Loading pattern {}: "{}"'.format(pattern_name, pattern))
                images.update({pattern_name: ImageDataset.load_files(base_path, pattern)})
        elif type(patterns) is list:
            for index, pattern in enumerate(patterns):
                logging.debug('Loading pattern {}: "{}"'.format(index, pattern))
                images.update({str(index): ImageDataset.load_files(base_path, pattern)})
        elif type(patterns) is str:
            logging.debug('Loading pattern {}: "{}"'.format(0, patterns))
            images.update({'str_patterns': ImageDataset.load_files(base_path, patterns)})
        return images

    @staticmethod
    def merge_patterns(patterns):
        images = []
        for pattern_name, files in patterns.items():
            images += files
        return images

    @staticmethod
    def load_files(base_path, pattern):
        found_images = []
        if type(pattern) is str:
            images = os.path.join(base_path, pattern)
            # search files
            found_images = glob.iglob(images)
        elif type(pattern) is list:
            for pattern in pattern:
                path = os.path.join(base_path, pattern)
                # search and add files
                found_images += glob.iglob(path)
        return found_images

    @staticmethod
    def image_info(patterns):
        for pattern_name, files in patterns.items():
            logging.info('Found {} {} data'.format(len(list(files)), pattern_name))

    def load_image(self, image):
        return self._load_image(image)

    def load_image_normal(self, image):
        image = Image.open(image)
        # apply Transforms
        if self.transform:
            image = self.transform(image)

        return image

    @lru_cache(maxsize=None)
    def load_image_cached(self, image):
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
