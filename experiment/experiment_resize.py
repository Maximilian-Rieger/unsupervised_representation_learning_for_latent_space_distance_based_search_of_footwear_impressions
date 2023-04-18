import logging
import re

import torch
import torch.nn.init
import torch.utils.data
import matplotlib as mpl
import os
import datetime
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from experiment.data_zoo import DataZoo
from utils.utils import dict_merge
import experiment.helper as helper

from dataloading.transforms import SlidingWindowTransformClass

mpl.use('Agg')


class Experiment:
    def __init__(self, args):
        self.args = {
            'log_dir': None,
            'config': None,
            'no_cuda': False,
            'cuda': True,
            'gpuid': 0,
            'name': None,

            'epochs': 1,
            'batchsize': 1,

            'img_size': 0,
            'channels': 0,
            'threshold': 0,
            'step': None,

            'training': {
                'data': None,

                'batchsize': 1,
                'shuffle': True,
                'worker': 0,
            },
        }
        changes = dict_merge(self.args, args)
        # forward arguments
        self.args = helper.forward_arguments(self.args, ['batchsize'], ['training'])
        self.loader_args = {'num_workers': 8, 'pin_memory': False} if self.args['cuda'] else {}
        if self.args['step'] is None:
            self.args['step'] = self.args['img_size']
        self.transform = self.get_transforms()
        self.resizing = None

    def get_transforms(self):
        crop_ratio = 0.95
        # jitter_range = (0., 1.)
        # jitter_range = (0.1, 0.9)
        jitter_range = (0.05, 0.95)
        rotation_range = 13 # (-30, 30)
        pre_crop_range = (0.95, 1)
        post_crop_range = (0.40, 7.0)
        return transforms.Compose([
            # transforms.CenterCrop((8400 * crop_ratio, 5100 * crop_ratio)),
            # transforms.ColorJitter(brightness=jitter_range, contrast=jitter_range, saturation=jitter_range),
            transforms.Resize(self.args['img_shape']),
            transforms.RandomResizedCrop(self.args['img_shape'], pre_crop_range),
            transforms.RandomOrder([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(rotation_range, fill=255),
            ]),
            # transforms.RandomChoice([
            #     transforms.Compose([
            #         transforms.RandomHorizontalFlip(),
            #         transforms.RandomChoice([
            #             transforms.RandomRotation(rotation_range, fill=255),
            #             transforms.RandomOrder([
            #                 # transforms.RandomResizedCrop(self.args['img_shape'], post_crop_range),
            #                 transforms.RandomRotation(rotation_range, fill=255),
            #             ]),
            #         ]),
            #         # transforms.ColorJitter(brightness=jitter_range, contrast=jitter_range, saturation=jitter_range)
            #     ]),
            #     # transforms.ColorJitter(brightness=jitter_range, contrast=jitter_range, saturation=jitter_range)
            #     # transforms.ColorJitter(brightness=(0.0, 0.5), contrast=(0.0, 0.5), saturation=(0.0, 0.5))
            # ]),
            # transforms.ColorJitter(contrast=jitter_range, saturation=jitter_range),
            # transforms.ColorJitter(saturation=jitter_range),
            transforms.ColorJitter(contrast=jitter_range),
            # transforms.ColorJitter(brightness=jitter_range),
            transforms.ToTensor(),
        ])

    def run(self):
        start = 0
        end = start + self.args['epochs']

        now = datetime.datetime.now()
        directory = '%d-%02d-%02d-%02d-%02d' % (now.year, now.month, now.day, now.hour, now.minute)
        exp_log_path = os.path.expanduser(self.args['log_dir'])
        exp_log_path = os.path.join(exp_log_path, directory)
        os.mkdir(exp_log_path)
        os.mkdir(os.path.join(exp_log_path, 'data'))

        # initialize data
        self.resizing = Rescale(**self.args['training'], transform=self.transform, log_dir=exp_log_path)

        for epoch in range(start, end):
            self.resizing(epoch=epoch)


class Rescale:
    global_step = 0

    def __init__(self, data, batchsize, shuffle, worker, log_dir, **kwargs):
        self.data = DataZoo.get(**data, **kwargs)
        self.log_dir = log_dir
        self.dataloader = DataLoader(self.data, batch_size=batchsize, shuffle=False, num_workers=worker)
        self.patches_saved_global = 0

    def __call__(self, epoch):
        pbar = tqdm(self.dataloader)
        pbar.set_description('Epoch {}'.format(epoch))
        patches_saved = 0
        for i, (paths, imgs) in enumerate(pbar):
            batches_done = epoch * len(self.data) + i
            for p in range(imgs.shape[0]):
                img = imgs[p,:,:,:]
                label = re.match(r'.*impress_(\d{1,3})', paths[p]).group(1)

                save_image(img, "{}/data/impress_{}_{}_{}_p{}.png".format(self.log_dir, label.zfill(3), epoch, batches_done, p), nrow=1, normalize=True)
                patches_saved += 1

            # log values
            Rescale.global_step += self.dataloader.batch_size
        self.patches_saved_global += patches_saved
        logging.info('Patching epoch {} patches saved {}'.format(epoch, self.patches_saved_global))

        return  # do training
