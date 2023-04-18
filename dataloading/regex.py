import torch.utils.data as data

from PIL import Image
import os
import os.path
from tqdm import tqdm
import numpy as np
import re
from PIL import ImageOps
import glob
from torch.utils.data import DataLoader
import logging

IMG_EXTENSIONS = (
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif', '.tiff', '.TIF', '.TIFF'
)


def is_image_file(filename, ext=IMG_EXTENSIONS):
    return any(filename.endswith(extension) for extension in ext)


def make_dataset(cur_dir, rxs, extensions):
    assert rxs is not None, 'no regular expression is set'
    cur_dir = os.path.expanduser(cur_dir)

    files = [f for f in tqdm(glob.glob(cur_dir + '/**/*.*', recursive=True), 'Parsing Filenames')
             if os.path.isfile(f) and is_image_file(f, extensions)]
    files.sort()

    if len(files) == 0:
        raise (RuntimeError("Found 0 data in subfolders of: {}\n"
                            "Supported image extensions are: {}".format(cur_dir, ",".join(extensions))))

    # this below should probably be moved to a regex label class or something
    # could be changed to imgage dataset and label decorator
    labels = {}
    label_to_int = {}
    int_to_label = {}
    for path in tqdm(files, 'Labels'):
        f = os.path.basename(path)
        for name, regex in rxs.items():
            r = '_'.join(re.search(regex, f).groups())
            labels[name] = labels.get(name, [])
            labels[name].append(r)

            label_to_int[name] = label_to_int.get(name, {})
            label_to_int[name][r] = label_to_int[name].get(r, len(label_to_int[name]))

            int_to_label[name] = int_to_label.get(name, {})
            int_to_label[name][label_to_int[name][r]] = r

    for name, lst in labels.items():
        labels[name] = [label_to_int[name][l] for l in lst]

    return files, labels, label_to_int, int_to_label


def pil_loader(path):
    # open jsonPath as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            if len(img.mode) > 1:
                return ImageOps.grayscale(img.convert('RGB'))

            return img.convert(mode='L')


def svg_string_loader(path):
    with open(path, 'r') as f:
        return f.read()


def get_loader(loader_name):

    if loader_name == 'svg_string':
        return svg_string_loader
    else:
        return pil_loader


class WrapableDataset(data.Dataset):

    def __len__(self):
        raise NotImplementedError

    @staticmethod
    def supported_classes():
        import dataloading.wrapper.triplets
        import dataloading.wrapper.pairs

        return {'CombineLabels': 'CombineLabels',
                'SelectLabels': 'SelectLabels',
                'TransformImages': 'TransformImages',
                'Sample': 'Sample',
                'CreateTriplets': dataloading.wrapper.triplets.CreateTriplets,
                'CreatePairs': dataloading.wrapper.pairs.CreatePairs}

    def _get_wrapper_class_constructor(self, name):
        def wrapper(*args, **kw):
            c = self.supported_classes()[name]
            if type(c) == str:
                return globals()[c](self, *args, **kw)
            else:
                return c(self, *args, **kw)

        return wrapper

    def __getattr__(self, attr):
        if attr in self.supported_classes():
            return self._get_wrapper_class_constructor(attr)

    def __getitem__(self, index):
        return self.get_image(index), self.get_label(index)


class DatasetWrapper(WrapableDataset):

    def __getattr__(self, attr):
        if attr in self.supported_classes():
            return self._get_wrapper_class_constructor(attr)
        else:
            return getattr(self.dataset, attr)

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)


class Sample(DatasetWrapper):
    def __init__(self, *args, samples=None, **kwargs):
        super().__init__(*args, **kwargs)

        if not samples:
            samples = len(self.dataset)

        self.samples = min(samples, len(self.dataset))  # we don't want to sample more than we have
        self.idx = np.linspace(0, len(self.dataset) - 1, self.samples, dtype=np.int32)

    def get_label(self, index):
        return self.dataset.get_label(self.idx[index])

    def get_image(self, index):
        return self.dataset.get_image(self.idx[index])

    def __len__(self):
        return len(self.idx)


class CombineLabels(DatasetWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.packed_labels = np.array([np.asscalar(np.argwhere(np.all(self.unique_labels == l, axis=1))) for l in
                                       self.packed_labels])
        self.unique_labels = np.unique(self.packed_labels, axis=0)

    def get_label(self, index):
        return self.packed_labels[index]


class SelectLabels(DatasetWrapper):
    def __init__(self, *args, label_names, **kwargs):
        super().__init__(*args, **kwargs)

        self.label_names = label_names if type(label_names) == list else [label_names]

        self.packed_labels = np.stack([self.labels[l] for l in self.label_names], axis=1)
        self.unique_labels = np.unique(self.packed_labels, axis=0)

    def get_label(self, index):
        label = tuple(self.packed_labels[index])

        if len(label) == 1:
            label = label[0]

        return label

    def __getitem__(self, index):
        return self.dataset.get_image(index), self.get_label(index)


class TransformImages(DatasetWrapper):
    def __init__(self, *args, transform, **kwargs):
        super().__init__(*args, **kwargs)
        self.transform = transform

    def get_image(self, index):
        img = self.dataset.get_image(index)

        # from torchvision import Transforms
        # from io import BytesIO
        # import cairosvg
        # target_size = 32
        #
        # def pad_img(img):
        #     delta_w = target_size - img.size[0]
        #     delta_h = target_size - img.size[1]
        #     padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        #     return ImageOps.expand(img, padding, fill=255)
        #
        # def svg_to_png(svg_str):
        #     rand = np.random.randint(10, 25) / 10.0
        #     svg_str = svg_str.replace('stroke-width="1"', 'stroke-width="{}"'.format(rand))
        #     img = Image.open(BytesIO(cairosvg.svg2png(svg_str)))
        #     return img.convert(mode='L')
        #
        # transform = Transforms.Compose([
        #     Transforms.Lambda(lambda x: svg_to_png(x)),
        #     Transforms.Lambda(lambda x: pad_img(x)),
        #     Transforms.Resize(target_size, interpolation=Image.NEAREST),
        # ])
        # transform(img).save('/caa/Homes01/fiel/tmp/patches/{}.png'.format(index))

        return self.transform(img)


class ImageFolder(WrapableDataset):
    """A generic data loader where the data are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory jsonPath.
        loader (callable, optional): A function to load an image given its jsonPath.

     Attributes:
        imgs (list): List of (image jsonPath, class_index) tuples
    """

    def __init__(self, path, loader='PIL', regex=None, mean=None, extensions=IMG_EXTENSIONS):
        logging.info('Loading dataset from {}'.format(path))
        # classes, class_to_idx = find_classes(root, regex)
        # imgs,classes, regex_to_class, indices = make_dataset(root, regex, id_regex)

        # this should be separated into imagefolderdataset and regexlabeldecorator
        # label are pretty independent and the standard implementation does not make much sense
        # however, regex kinda belongs to the image folder dataset since this depends on the filenames and dataset
        # so probably it is fine as it is ...
        imgs, labels, label_to_int, int_to_label = make_dataset(path, regex, extensions)

        self.label_to_int = label_to_int
        self.int_to_label = int_to_label
        self.label_names = [name for name, _ in regex.items()]
        self.packed_labels = np.stack([labels[l] for l in self.label_names], axis=1)

        self.labels = labels
        self.root = path
        self.imgs = imgs
        self.loader = get_loader(loader)
        self.regex = regex
        self._mean = mean

    @property
    def mean(self):
        if type(self._mean) == str:
            self._mean = np.load(os.path.join(self.root, self._mean))
        elif self._mean is None:
            cur_data = DataLoader(self, batch_size=1000, shuffle=False, num_workers=8)

            mean = None
            logging.info('Calculating mean image for "{}"'.format(self.root))

            cnt = 0
            for img, _ in tqdm(cur_data, 'Calculating Mean'):
                s = img.size(0)
                m = np.mean(img.numpy(), axis=0)

                if mean is None:
                    mean = m
                else:
                    mean = mean + (m - mean) * s / (s + cnt)

                cnt += s
            self._mean = mean
            # np.save(os.jsonPath.join(self.root, 'mean.npy'), self._mean, allow_pickle=False)
        return self._mean

    def get_image(self, index):
        img = self.imgs[index]
        img = self.loader(img)
        return img

    def get_label(self, index):
        label = tuple(self.packed_labels[index])

        if len(label) == 1:
            label = label[0]

        return label

    def __len__(self):
        return len(self.imgs)
