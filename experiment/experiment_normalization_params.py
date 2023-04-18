import logging
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

# mean = torch.zeros(3)
# std = torch.zeros(3)
#
# for i, data in enumerate(dataloader):
#     if (i % 10000 == 0): print(i)
#     data = data[0].squeeze(0)
#     if (i == 0): size = data.size(1) * data.size(2)
#     mean += data.sum((1, 2)) / size
#
# mean /= len(dataloader)
# print(mean)
# mean = mean.unsqueeze(1).unsqueeze(2)
#
# for i, data in enumerate(dataloader):
#     if (i % 10000 == 0): print(i)
#     data = data[0].squeeze(0)
#     std += ((data - mean) ** 2).sum((1, 2)) / size
#
# std /= len(dataloader)
# std = std.sqrt()
# print(std)


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
        crop_ratio = 0.90
        return transforms.Compose([
            transforms.CenterCrop((8400 * crop_ratio, 5100 * crop_ratio)),
            transforms.Resize(self.args['img_shape']),
            # transforms.RandomResizedCrop(self.args['img_shape'], (0.8, 1)),
            # transforms.RandomChoice([
            #     transforms.RandomResizedCrop(self.args['img_shape'], (0.8, 0.9)),
            #     transforms.Compose([
            #         transforms.RandomOrder([
            #             transforms.RandomHorizontalFlip(),
            #             transforms.RandomRotation(30, fill=255)
            #         ]),
            #         transforms.ColorJitter(brightness=(0.3, 0.8), contrast=(0.3, 0.8)),
            #     ]),
            #     transforms.Compose([
            #         transforms.RandomOrder([
            #             transforms.RandomResizedCrop(self.args['img_shape'], (0.8, 9.0)),
            #             transforms.RandomRotation(30, fill=255),
            #         ]),
            #         transforms.ColorJitter(brightness=(0.3, 0.8), contrast=(0.3, 0.8)),
            #     ]),
            #     transforms.ColorJitter(brightness=(0.2, 0.8), contrast=(0.3, 0.8), saturation=(0.3, 0.8))
            # ]),
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
        self.dataloader = DataLoader(self.data, batch_size=batchsize, shuffle=True, num_workers=worker)
        self.patches_saved_global = 0

    def __call__(self, epoch):
        pbar = tqdm(self.dataloader)
        pbar.set_description('Epoch {}'.format(epoch))
        patches_saved = 0
        for i, imgs in enumerate(pbar):
            batches_done = epoch * len(self.data) + i

            for p in range(imgs.shape[0]):
                img = imgs[p,:,:,:]
                save_image(img, "{}/data/patch_{}_{}_{}.png".format(self.log_dir, epoch, batches_done, p), nrow=1, normalize=True)
                patches_saved += 1

            # log values
            Rescale.global_step += self.dataloader.batch_size
        self.patches_saved_global += patches_saved
        logging.info('Patching epoch {} patches saved {}'.format(epoch, self.patches_saved_global))

        return  # do training
