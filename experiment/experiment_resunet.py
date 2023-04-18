import os
import datetime
import numpy as np
import logging
from pathlib import Path
import torch
from torch import nn
import torch.nn.init
import torch.utils.data
from torch.optim import lr_scheduler
import matplotlib as mpl
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sys import exit
from experiment.data_zoo import DataZoo
from torchvision.utils import save_image, make_grid

from utils.utils import TensorboardLogger
from utils.utils import dict_merge
from utils.utils import GPU
from utils.keyboard_menu import KeyboardMenu

import experiment.helper as helper

from models.ResUNet import ResUNet
from experiment.helper import MaskToTensor

from optimizers.RAdam import RAdam

mpl.use('Agg')


class Experiment:
    def __init__(self, args):
        self.args = {
            'log_dir': None,
            'config': None,
            'resume': None,
            'no_cuda': False,
            'cuda': True,
            'gpuid': 0,
            'name': None,

            'start_epoch': 0,
            'epochs': 1,
            'batchsize': 1,

            'img_size': 128,
            'channels': 128,

            'optimizer': {
                'lr': 0.0001,
                'step_size': 1,
                'beta1': 0.99,
                'beta2': 0.999,
                'gamma': 0.1
            },

            'training': {
                'data': None,

                'batchsize': 1,
                'shuffle': True,
                'worker': 0,
                'n_checkpoints': None,
                'sample_interval': None,
            },
        }

        self.keyboard_menu = None
        self.image_transform = None
        self.label_transform = None

        # check config by comparing with defaults and merge
        changes = dict_merge(self.args, args, verify=True)

        # forward arguments
        self.args = helper.forward_arguments(self.args, ['batchsize'], sets=['training'])

        self.logger = TensorboardLogger(os.path.join(args['log_dir'], args['name']), modules=[__name__], images_dir=True)

        now = datetime.datetime.now()
        self.args['started_training'] = '%d-%02d-%02d-%02d-%02d' % (now.year, now.month, now.day, now.hour, now.minute)
        self.logger.log_options(self.args, changes)

        self.loader_args = {'num_workers': 8, 'pin_memory': False} if self.args['cuda'] else {}

    def __str__(self) -> str:
        return '{}x{}x{}' \
            .format(self.args['batchsize'], self.args['in'], self.args['out'])

    @staticmethod
    def model() -> [ResUNet, nn.Module]:
        model = ResUNet(
            in_channels=3,
            n_classes=1,
            depth=5,  # new setting
            last=nn.Tanh
        )

        model.to(GPU.device)
        criterion = nn.L1Loss().to(GPU.device)

        return [model, criterion]

    @staticmethod
    def optimizer(model, lr, step_size, beta1=0.9, beta2=0.99, gamma=0.1):
        optimizer = RAdam(model.parameters(), lr=lr, betas=(beta1, beta2))

        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        return [optimizer, scheduler]

    def setup(self):
        in_size = (self.args['img_size'], self.args['img_size'])

        self.shoe_transform = transforms.Compose([
            transforms.Resize(in_size),
            transforms.ToTensor(),
        ])

        self.sole_transform = transforms.Compose([
            transforms.Resize(in_size),
            transforms.RandomRotation((90, +90)),
            transforms.Grayscale(),
            MaskToTensor(),
        ])

        self.transforms = {
            'image': self.shoe_transform,
            'impression': self.sole_transform,
        }

        # initialize training data
        self.training = Training(**self.args['training'], logger=self.logger, transforms=self.transforms)

        model, criterion = Experiment.model()
        optimizer, scheduler = Experiment.optimizer(model, **self.args['optimizer'])

        if self.args['resume'] is not None:
            model, optimizer, train_step = helper.load_checkpoint(model, optimizer)
            Training.global_step = train_step

        return model, criterion, optimizer, scheduler

    def run(self):
        model_ft, criterion, optimizer, scheduler = self.setup()

        def abort():
            helper.save_checkpoint(epoch, self.logger.global_step, model_ft, optimizer, self.logger.log_dir, None, True)
            exit()

        self.keyboard_menu = KeyboardMenu(abort_method=abort)

        start = self.args['start_epoch']
        end = start + self.args['epochs']
        Path(os.path.join(self.logger.log_dir, 'started_trainings_loop.txt')).touch()

        for epoch in range(start, end + 1):
            self.logger.epoch = epoch

            self.training(model=model_ft, criterion=criterion, epoch=epoch, logger=self.logger, menu=self.keyboard_menu,
                          optimizer=optimizer, scheduler=scheduler)


class Training:
    global_step = 0

    def __init__(self, data, batchsize, shuffle, worker, logger, sample_interval, n_checkpoints, **kwargs):
        self.data = DataZoo.get(**data, **kwargs)
        self.dataloader = DataLoader(self.data, batch_size=batchsize, shuffle=shuffle, num_workers=worker)
        self.sample_interval = sample_interval
        self.n_checkpoints = n_checkpoints
        self.logger = logger

    def __call__(self, model, criterion, epoch, logger, menu, optimizer, scheduler):
        model.train()
        scheduler.step()
        total_train = 0
        correct_train = 0
        running_loss = 0.0
        running_accuracy = 0.0
        # running_correct_ratio = 0.0

        pbar = tqdm(self.dataloader)
        pbar.set_description('Epoch {}'.format(epoch))
        for i, [_input, target] in enumerate(pbar):
            _input = _input.to(GPU.device)
            output = model(_input)
            target = target.to(GPU.device)
            target = torch.unsqueeze(target, 1)

            loss = criterion(output, target)

            # compute gradient and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log values
            logger.step(self.dataloader.batch_size)
            Training.global_step += self.dataloader.batch_size

            # calculate log values
            running_loss += loss.item() * _input.size(0)

            logger.log_value('train_loss', loss.item(), Training.global_step)

            predicted = torch.max(output, 1)[1]
            total_train += target.nelement()

            correct_train += predicted.eq(target.to(GPU.device).type(torch.int64).detach()).sum().item()
            # running_correct_ratio += correct_train / predicted.shape
            train_accuracy = 100 * correct_train / total_train
            logger.log_value('train_accuracy', train_accuracy, Training.global_step)
            running_accuracy += train_accuracy
            batches_done = epoch * len(self.data) + i
            if batches_done % self.sample_interval == 0:
                self.sample_image(model, n_row=self.dataloader.batch_size, batches_done=batches_done)

            # menu()

        # calculate log values
        epoch_loss = running_loss / len(self.data)
        epoch_accuracy = running_accuracy / len(self.data)
        epoch_accuracy = running_accuracy / len(self.data)

        logger.log_value('train_epoch_loss', epoch_loss, Training.global_step)
        logging.info('Training {} epoch {} Loss: {:.4f}, Accuracy {}'
                     .format('Impress_ResUNet', epoch, epoch_loss, epoch_accuracy))
        # do checkpointing
        if self.n_checkpoints is None or epoch % self.n_checkpoints == 0:
            helper.save_checkpoint(epoch, Training.global_step, [model], [optimizer], logger.log_dir, self.n_checkpoints)
        return  # do training

    def sample_image(self, model, n_row, batches_done):
        """Saves a grid of generated data"""

        batch_size = self.dataloader.batch_size
        batch_start = ((batches_done * batch_size) % len(self.data))
        batch_end = ((batches_done * batch_size + batch_size) % len(self.data))

        shoes, soles = [], []
        for pair in self.data[batch_start:batch_end]:
            shoe, sole = pair
            shoes += [shoe]
            soles += [sole]
        sole = torch.stack(soles)
        sole = sole.to(GPU.device)
        shoe = torch.stack(shoes)
        shoe = shoe.to(GPU.device)

        batch = model(shoe.detach())
        batch_grid = make_grid(batch.detach(), nrow=n_row, normalize=True)
        input_grid = make_grid(shoe, nrow=n_row, normalize=True)
        sole_grid = make_grid(sole, nrow=n_row, normalize=True)

        batch_stack = torch.stack([input_grid, batch_grid])
        img_grid = make_grid(batch_stack, nrow=n_row, normalize=True)
        fig, subplots = helper.create_image_figure(img_grid.cpu(), 'Transformed', 1, 2)

        img_grid = make_grid(sole_grid, nrow=n_row, normalize=True)
        img_grid = torch.unsqueeze(img_grid, 1)
        img_grid = make_grid(img_grid, nrow=n_row, normalize=True)
        subplots[1].imshow(np.moveaxis(img_grid.cpu().numpy(), 0, -1))
        subplots[1].set_title('Target')
        self.logger.add_figure('{}_sample_{:0>10}'.format('Impress', batches_done), fig)
        fig.savefig("{}/data/sample_{}.fig.png".format(self.logger.log_dir, batches_done))
        save_image(img_grid, "{}/data/sample_{}.png".format(self.logger.log_dir, batches_done), nrow=n_row, normalize=True)

