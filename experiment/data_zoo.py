import os.path

import torch
import torchvision

from dataloading.imagedataset import ImageDataset
from dataloading.impressdataset import ImpressDataset
from dataloading.impress_aligned_dataset import ImpressAlignedDataset
from dataloading.impress_aligned_dataset_clean import ImpressAlignedDatasetClean
from dataloading.cityscapes_imagedataset import CityScapesImageDataset
from dataloading.patchdataset import PatchDataset
from dataloading.deaugmentingimpressdataset import DeAugmentingImpressDataset

import re

from experiment.mnist_tests import CachedDataset


class DataZoo:
    @classmethod
    def get(cls, base, dataset, set, **kwargs):
        _all = cls.datasets
        d = _all[dataset]
        if 'args' in d:
            kwargs = {**d['args'], **kwargs}

        base = os.path.expanduser(base)

        if dataset == 'Impress_combined':
            return ImpressDataset(
                sole_base_path=os.path.join(base, _all['Impress_soles']['basepath']),
                shoe_base_path=os.path.join(base, _all['Impress']['basepath']),
                limit=d['limit'] if 'limit' in d else None, **kwargs)
        elif dataset == 'Impress_registrered':
            return ImpressAlignedDataset(base_path=os.path.join(base, d['basepath']), limit=d['limit'] if 'limit' in d else None, **kwargs)
        elif dataset == 'Impress_registrered_clean':
            return ImpressAlignedDatasetClean(base_path=os.path.join(base, d['basepath']), limit=d['limit'] if 'limit' in d else None, **kwargs)
        # elif dataset == 'CelebA': # should use normal ImageDataset
        #     return CelebAImageDataset(base_path=os.path.join(base, d['basepath']), limit=d['limit'] if 'limit' in d else None, **kwargs)
        elif dataset == 'CITYSCAPES':
            return DataZoo.cityscapes(base, dataset, set, **kwargs)
        elif dataset == 'Patches':
            return PatchDataset(base_path=os.path.join(base, d['basepath'], d['set'][set]['path']), limit=d['limit'] if 'limit' in d else None, **kwargs)
        elif dataset == 'Patches_512':
            return PatchDataset(base_path=os.path.join(base, d['basepath'], d['set'][set]['path']), limit=d['limit'] if 'limit' in d else None, **kwargs)
        elif dataset == 'Patches_Extended':
            return PatchDataset(base_path=os.path.join(base, d['basepath'], d['set'][set]['path']), **kwargs)
        elif dataset == 'Impress_DeAugment':
            return DeAugmentingImpressDataset(base_path=os.path.join(base, d['basepath']), **kwargs)
        elif dataset == 'MNIST':
            # trainset = CachedDataset(torchvision.datasets.MNIST(root='S:\Data\Datasets', train=True, download=True, transform=transform))
            if set == 'train':
                return CachedDataset(torchvision.datasets.MNIST(root='S:\Data\Datasets', train=True, download=True, transform=kwargs['transform']))
            elif set == 'test':
                return CachedDataset(torchvision.datasets.MNIST(root='S:\Data\Datasets', train=False, download=True, transform=kwargs['transform']))

        return ImageDataset(base_path=os.path.join(base, d['basepath']), set=set, **kwargs)

    @classmethod
    def cityscapes(cls, base, dataset, set, **kwargs):
        _all = cls.datasets
        d = _all[dataset]
        s = d['set'][set]

        base = os.path.expanduser(base)
        return CityScapesImageDataset(base_path=os.path.join(base, d['basepath']),
                            set=set,
                            image_dir=d['images'],
                            limit=s['limit'] if 'limit' in s else None,
                            **kwargs
                            )

    datasets = {
        'MNIST': {
        },
        'Patches': {
            'basepath': 'patches_full',
            'set': {
                'training': {'path': 'training'},
                'validation': {'path': 'validation'},
            }
        },
        'Patches_512': {
            'basepath': 'patches_full_512',
            'set': {
                'training': {'path': 'training'},
                'validation': {'path': 'validation'},
            }
        },
        'Patches_extended': {
            'basepath': 'patches_extended',
            'set': {
                'training': {'path': 'training'},
                'validation': {'path': 'validation'},
            }
        },
        'Impress': {
            'basepath': 'impress schuhe+spezial',
        },
        'Impress_soles': {
            'basepath': 'impress',
            'args': {
                'pattern': {
                    'left': '*_3_L.jpg',
                    'right': '*_1_R.jpg'
                }
            }
        },
        'Impress_soles_extra': {
            'basepath': 'impress',
            'args': {
                'pattern': {
                    'rough-right': ['*_a_R.jpg', '*_b_R.jpg'],
                    'rough-left': ['*_c_L.jpg', '*_d_L.jpg'],
                }
            }
        },
        'Impress_soles_background': {
            'basepath': 'impress',
            'args': {
                'pattern': {
                    'right-with-background': '*_c_R.jpg',
                }
            }
        },
        'Impress_soles_full': {
            'basepath': 'sorted-inkless',

            'args': {
                'pattern': {
                    'left': '*_3_3.jpg',
                    'right': '*_3_1.jpg'
                },
                'shared_pattern': '*',
            }
        },
        'Impress_soles_extended': {
            'basepath': 'sorted-inkless',
            'args': {
                'pattern': {
                    'left': ['*_3_3.jpg', '*_*_L.jpg'],
                    'right': ['*_3_1.jpg', '*_*_R.jpg'],
                }
            }
        },
        'Impress_soles_prescaled': {
            'basepath': 'impress_prescaled',
            'args': {
                'pattern': {
                    'patches': ['patch_*_*_*.png'],
                },
                'shared_pattern': '',
            }
        },
        'Impress_soles_prescaled_2': {
            'basepath': 'impress_prescaled_2',
            'args': {
                'pattern': {
                    'patches': ['patch_*_*_*.png'],
                },
            }
        },
        'Impress_cleaned': {
            'basepath': 'impress_cleaned',
            'args': {
                'pattern': {
                    'patches': ['impress_*_*_*_*.png'],
                },
            }
        },
        'Impress_cleaned_2': {
            'basepath': 'impress_cleaned_3',
            'args': {
                'pattern': {
                    'patches': ['impress_*_*_*_*.png'],
                },
            }
        },
        'Impress_2': {
            'basepath': 'impress_2',
            'args': {
                'pattern': {
                    'patches': ['impress_*_*_*_*.png'],
                },
                'path_transform': lambda path: int(re.match(r'.*impress_(\d{1,3})', path).group(1)),
                # 'limit': 128
            },
        },
        'Impress_DeAugment': {
            'basepath': 'impress_2',
            'args': {
                'pattern': {
                    'patches': ['impress_*_*_*_*.png'],
                },
                'path_transform': lambda path: int(re.match(r'.*impress_(\d{1,3})', path).group(1))
            }
        },
        'FID-300': {
            'basepath': 'FID-300/references',
            'args': {
                'pattern': {
                    'patches': ['*.png'],
                },
                'shared_pattern': '',
            }
        },
        'Impress_cleaned_prescaled': {
            'basepath': 'impress_prescaled_3',
            'args': {
                'pattern': {
                    'patches': ['impress_*_*_*_*.png'],
                },
            }
        },
        'Impress_combined': {},
        'Impress_registrered': {
            'basepath': 'impress_registered',
        },
        'Impress_registrered_clean': {
            'basepath': 'impress_registered_clean',
        },
        'CelebA': {
            'basepath': 'img_align_celeba',
            'args': {
                'pattern': {
                    'faces': ['*.jpg'],
                },
                'shared_pattern': '',
                'limit': 25_000,
                # 'limit': 128,
            },
            # 'limit': 1000
        },
        'CelebA-val': {
            'basepath': 'img_align_celeba',
            'args': {
                'pattern': {
                    'faces': ['*.jpg'],
                },
                'shared_pattern': '',
                'limit': 51_024,
                'offset': 50_000,
            },
        },
        'CITYSCAPES': {
            'basepath': 'Cityscapes',
            'images': 'leftImg8bit',
            'set': {
                'train': {'labels': 'gtFine/train'},
                'test': {'labels': 'gtFine/test'},
                'val': {'labels': 'gtFine/test'},
            }
        },
        'CAT': {
            'basepath': 'cat_dataset',
            'args': {
                'pattern': {
                    'cats': ['CAT_*\\*_*.jpg'],
                },
                'shared_pattern': 'cats',
            }
        },
        'CAT-VAL': {
            'basepath': 'cat_dataset',
            'args': {
                'pattern': {
                    'cats': ['CAT_*\\*_*.jpg'],
                },
                'shared_pattern': 'cats_validation',
            }
        },
    }
