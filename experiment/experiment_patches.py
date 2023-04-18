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
        self.transform = transforms.Compose([
            transforms.Resize(self.args['img_pre_size']),
            transforms.ToTensor(),
            SlidingWindowTransformClass(self.args['img_size'], self.args['step'], self.args['threshold']),
        ])
        self.patching = None

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
        self.patching = Patching(**self.args['training'], transform=self.transform, log_dir=exp_log_path)

        for epoch in range(start, end):
            self.patching(epoch=epoch)


class Patching:
    global_step = 0

    def __init__(self, data, batchsize, shuffle, worker, log_dir, **kwargs):
        self.data = DataZoo.get(**data, **kwargs)
        self.log_dir = log_dir
        self.dataloader = DataLoader(self.data, batch_size=batchsize, shuffle=shuffle, num_workers=worker)

    def __call__(self, epoch):
        pbar = tqdm(self.dataloader)
        pbar.set_description('Epoch {}'.format(epoch))
        patches_saved = 0
        for i, imgs in enumerate(pbar):
            batches_done = epoch * len(self.data) + i

            for p in range(len(imgs[0,:,0,0])):
                img = imgs[0,p,:,:]
                save_image(img, "{}/data/patch_{}_{}_{}.png".format(self.log_dir, epoch, batches_done, p), nrow=1, normalize=True)
                patches_saved += 1

            # log values
            Patching.global_step += self.dataloader.batch_size

        logging.info('Patching epoch {} patches saved {}'.format(epoch, patches_saved))

        return  # do training
