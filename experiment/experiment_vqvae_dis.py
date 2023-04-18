import os
import asyncio
import datetime
from typing import Tuple, List

import itertools
import numpy as np
import logging
from pathlib import Path
import torch
from torch import nn, Tensor
import torch.nn.init
import torch.utils.data
from torch.optim import lr_scheduler
from torch.autograd import Variable
import matplotlib as mpl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from sys import exit
from data_zoo import DataZoo

from utils.utils import TensorboardLogger
from utils.utils import dict_merge
from utils.utils import GPU
from utils.keyboard_menu import KeyboardMenu

import experiment.helper as helper
from torchsummary import summary

from models.SupRVAE import Discriminator
from models.vqvae.vqvae import VQVAE
from optimizers.AdaBelief import AdaBelief

mpl.use('Agg')

SetupStruct = Tuple[
    List[nn.Module],
    List[nn.Module],
    List[torch.optim.Optimizer],
    List[torch.optim.lr_scheduler._LRScheduler]
]


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
            'scatter_plot_interval': None,

            'model': {
                'n_hiddens': 128,
                'n_residual_hiddens': 32,
                'n_residual_layers': 2,
                'embedding_dim': 64,
                'n_embeddings': 512,
                'beta': 0.25,
            },

            'img_size': 0,
            'channels': 0,

            'optimizer': {
                'lr': 0.01,
                'step_size': 5,
                'beta1': 0.9,
                'beta2': 0.99,
                'gamma': 0.5
            },

            'training': {
                'data': None,

                'batchsize': 1,
                'shuffle': True,
                'worker': 0,
                'sample_interval': None,
                'n_checkpoints': None,
                'grad_vis': False
            },

            'validation': {
                'data': [None],

                'batchsize': 1,
                'worker': 0,
                'plot': None,
            }
        }

        self.keyboard_menu = None
        self.image_transform = None
        self.label_transform = None

        # check config by comparing with defaults and merge
        changes = dict_merge(self.args, args, verify=True)

        # forward arguments
        self.args = helper.forward_arguments(self.args, ['batchsize'], ['training'])

        self.logger = TensorboardLogger(os.path.join(args['log_dir'], args['name']), modules=[__name__],
                                        images_dir=True)

        now = datetime.datetime.now()
        self.args['started_training'] = '%d-%02d-%02d-%02d-%02d' % (now.year, now.month, now.day, now.hour, now.minute)
        self.logger.log_options(self.args, changes)

        self.loader_args = {'num_workers': 8, 'pin_memory': False} if self.args['cuda'] else {}

    def __str__(self) -> str:
        return '{}x{}x{}' \
            .format(self.args['batchsize'], self.args['in'], self.args['out'])

    @staticmethod
    def model(in_chan, n_hiddens, n_residual_hiddens, n_residual_layers, embedding_dim, n_embeddings, beta) -> [[nn.Module], [nn.Module]]:
        activation = nn.ReLU

        vqvae = VQVAE(in_chan, n_hiddens, n_residual_hiddens, n_residual_layers, embedding_dim, n_embeddings, beta).to(GPU.device)
        discriminator = Discriminator(embedding_dim, depth=n_residual_layers, activation=activation).to(GPU.device)

        summary(vqvae.encoder, input_size=(vqvae.encoder.in_dim, 256, 256), batch_size=8)
        summary(vqvae.decoder, input_size=(vqvae.decoder.in_dim, embedding_dim, embedding_dim), batch_size=8)
        summary(discriminator, input_size=(embedding_dim,), batch_size=8)

        recon_loss = torch.nn.MSELoss().to(GPU.device)

        discriminator_loss = torch.nn.BCELoss().to(GPU.device)

        return [[vqvae, discriminator], [recon_loss, discriminator_loss]]

    @staticmethod
    def optimizer(models, lr, step_size, beta1=0.9, beta2=0.99, gamma=0.1):
        vqvae, discriminator = models

        optimizer_vqvae = AdaBelief(vqvae.parameters(), lr=lr, betas=(beta1, beta2))
        optimizer_Dis = AdaBelief(discriminator.parameters(), lr=lr, betas=(beta1, beta2))


        scheduler_vqvae = lr_scheduler.StepLR(optimizer_vqvae, step_size=step_size, gamma=gamma)
        scheduler_Dis = lr_scheduler.StepLR(optimizer_Dis, step_size=step_size, gamma=gamma)

        return [[optimizer_vqvae, optimizer_Dis], [scheduler_vqvae, scheduler_Dis]]

    @staticmethod
    def get_img_shape(args, padding=0):
        if not isinstance(args['img_size'], list):
            return args['channels'], args['img_size'] + padding, args['img_size'] + padding
        else:
            return args['channels'], args['img_size'][0] + padding, args['img_size'][1] + padding

    def setup(self) -> SetupStruct:
        img_shape = self.get_img_shape(self.args)

        self.transform = transforms.Compose([
            transforms.Resize(img_shape[1:]),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

        # initialize training data
        self.training = Training(**self.args['training'], transform=self.transform, logger=self.logger)

        models, losses = Experiment.model(self.args['channels'], **self.args['model'])
        optimizers, schedulers = Experiment.optimizer(models, **self.args['optimizer'])

        if self.args['resume'] is not None:
            models, optimizers, train_step = helper.load_checkpoint(models, optimizers)
            Training.global_step = train_step

        return models, losses, optimizers, schedulers

    def run(self):
        setup = self.setup()

        def abort():
            helper.save_checkpoint(epoch, self.logger.global_step, [setup[0]], [setup[2]], self.logger.log_dir, None, True)
            exit()

        self.keyboard_menu = KeyboardMenu(abort_method=abort)

        start = self.args['start_epoch']
        end = start + self.args['epochs']
        Path(os.path.join(self.logger.log_dir, 'started_trainings_loop.txt')).touch()
        # torch.autograd.set_detect_anomaly(True)

        for epoch in range(start, end + 1):
            self.logger.epoch = epoch
            self.training(setup=setup, epoch=epoch, menu=self.keyboard_menu)

        helper.save_checkpoint(end + 1, Training.global_step, setup[0], setup[2], self.logger.log_dir, self.args['training']['n_checkpoints'])


class Training:
    global_step = 0

    def __init__(self, data, batchsize, shuffle, worker, sample_interval, n_checkpoints, logger, grad_vis, **kwargs):
        self.data = DataZoo.get(**data, **kwargs)
        self.dataloader = DataLoader(self.data, batch_size=batchsize, shuffle=shuffle, num_workers=worker)

        self.logger = logger
        self.sample_interval = sample_interval
        self.n_checkpoints = n_checkpoints
        self.epoch = None
        self.grad_vis = grad_vis
        self.grad_vis_g_done = not grad_vis

    def train(self, models, losses, optimizers, imgs, gt_vec):
        # Configure x
        real_imgs = Variable(imgs.type(Tensor).to(GPU.device))

        vqvae, discriminator = models
        mse_loss, gmsd_loss, discriminator_loss = losses
        optimizer_vqvae, optimizer_dis = optimizers
        valid, fake = gt_vec

        # ----------------------
        #  Train Reconstruction
        # ----------------------

        optimizer_vqvae.zero_grad()

        embedding_loss, x_hat, _, encoded_imgs = vqvae(real_imgs)
        embedding_loss = embedding_loss.mean()
        recon_loss = mse_loss(real_imgs, x_hat)
        extra_recon_loss = gmsd_loss(x_hat, real_imgs)
        forging_loss = discriminator_loss(discriminator(encoded_imgs), valid)
        self.logger.log_value_and_epoch_avg('embedding_loss', embedding_loss, self.epoch, self.dataloader.batch_size)
        self.logger.log_value_and_epoch_avg('forging_loss', forging_loss, self.epoch, self.dataloader.batch_size)
        self.logger.log_value_and_epoch_avg('recon_loss', recon_loss, self.epoch, self.dataloader.batch_size)
        self.logger.log_value_and_epoch_avg('gmsd_loss', gmsd_loss, self.epoch, self.dataloader.batch_size)

        # loss = adv_weight * embedding_loss + (1 - adv_weight) * recon_loss
        loss = embedding_loss + recon_loss + extra_recon_loss + forging_loss

        loss.backward()
        optimizer_vqvae.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_dis.zero_grad()

        z = torch.Tensor(np.random.normal(0, 1, (self.dataloader.batch_size, vqvae.embedding_dim))).to(GPU.device)
        z, _, _ = vqvae.vector_quantization.strait_through(z).detach()

        pred = discriminator(z)
        # Calculate accuracy for fake distributions
        accuracy_fake = self.accuracy(pred, fake)
        fake_loss = discriminator_loss(pred, fake)

        pred = discriminator(encoded_imgs.detach())

        # Calculate accuracy for real distributions
        accuracy_valid = self.accuracy(pred, valid)
        real_loss = discriminator_loss(pred, valid)

        d_loss = 0.5 * (real_loss + fake_loss)
        accuracy = 0.5 * (accuracy_fake + accuracy_valid)

        d_loss.backward()
        optimizer_dis.step()

        if not self.grad_vis_g_done:
            helper.plot_grad_flow_lines(vqvae.named_parameters(), f"{self.logger.log_dir}/data/Grad_flow_vqvae_{self.epoch}.lines.png")
            self.grad_vis_g_done = True

        return loss, d_loss, accuracy

    def accuracy(self, pred, target):
        return (pred > 0.5).type(torch.cuda.FloatTensor).eq(target).sum().item() / self.dataloader.batch_size

    def __call__(self, setup: SetupStruct, epoch: int, menu):
        models, losses, optimizers, schedulers = setup
        [model.train() for model in models]
        scheduler_vqvae, scheduler_dis = schedulers

        self.epoch = epoch
        self.grad_vis_g_done = not self.grad_vis
        self.grad_vis_d_done = not self.grad_vis

        self.logger.log_value('lr_vqvae', scheduler_vqvae.get_last_lr()[0], Training.global_step)
        self.logger.log_value('lr_vqvae_dis', scheduler_dis.get_last_lr()[0], Training.global_step)

        running_loss = 0.0
        running_d_loss = 0.0
        running_acc = 0.0

        # Adversarial ground truths
        valid = torch.Tensor(self.dataloader.batch_size, 1).to(GPU.device).fill_(1.0).requires_grad_(False)
        fake = torch.Tensor(self.dataloader.batch_size, 1).to(GPU.device).fill_(0.0).requires_grad_(False)

        pbar = tqdm(self.dataloader)
        pbar.set_description('Epoch {}'.format(epoch))

        for i, imgs in enumerate(pbar):
            batches_done = epoch * len(self.data) + i
            loss, d_loss, accuracy = self.train(models, losses, optimizers, imgs, [valid, fake])

            if self.sample_interval is not None and batches_done % self.sample_interval == 0:
                asyncio.run(self.sample_image(models, n_row=self.dataloader.batch_size, batches_done=batches_done))

            # log values
            self.logger.step(self.dataloader.batch_size)
            Training.global_step += self.dataloader.batch_size

            # calculate log values
            running_loss += loss.item() * self.dataloader.batch_size
            running_d_loss += d_loss.item() * self.dataloader.batch_size
            running_acc += accuracy * self.dataloader.batch_size

            self.logger.log_value('train_loss', loss.item(), Training.global_step)
            self.logger.log_value('train_d_loss', d_loss.item(), Training.global_step)
            self.logger.log_value('train_accuracy', accuracy, Training.global_step)

        # calculate log values
        epoch_loss = running_loss / len(self.data)
        epoch_d_loss = running_d_loss / len(self.data)
        epoch_acc = running_acc / len(self.data)

        self.logger.log_value('train_epoch_loss', epoch_loss, Training.global_step)
        self.logger.log_value('train_epoch_d_loss', epoch_d_loss, Training.global_step)
        self.logger.log_value('train_epoch_accuracy', epoch_acc, Training.global_step)
        logging.info(f'Training {"Impress_VQ-VAE"} epoch {epoch} Lr: {scheduler_vqvae.get_last_lr()[0]:.4f} Loss: {epoch_loss:.4f} D_Loss: {epoch_d_loss:.4f}')
        scheduler_vqvae.step()
        scheduler_dis.step()

        # do checkpointing
        if self.n_checkpoints is None or epoch % self.n_checkpoints == 0:
            helper.save_checkpoint(epoch, Training.global_step, models, optimizers, self.logger.log_dir, self.n_checkpoints)

        return  # do training

    async def sample_image(self, models, n_row, batches_done):
        """Saves a grid of generated data"""
        [model.eval() for model in models]
        with torch.no_grad():
            vqvae, = models

            batch_size = self.dataloader.batch_size

            batch_start = ((batches_done * batch_size) % len(self.data))
            batch_end = ((batches_done * batch_size + batch_size) % len(self.data))

            imgs = self.data[batch_start:batch_end]
            imgs = torch.stack(imgs)
            imgs = imgs.to(GPU.device)

            batch = vqvae(imgs.detach())

            batch_grid = make_grid(batch.detach(), nrow=n_row // 2)
            input_grid = make_grid(imgs, nrow=n_row // 2)

            batch_stack = torch.stack([input_grid, batch_grid])
            img_grid = make_grid(batch_stack, nrow=1)
            img_grid = helper.normalize(img_grid)
            save_image(img_grid.data, f"{self.logger.log_dir}/data/Reconstruction_{batches_done}.png", nrow=n_row)
            fig, subplots = helper.create_image_figure(img_grid.cpu(), 'Reconstructed')
            self.logger.add_figure(f'Impress_VQ-VAE_Reconstruction_{batches_done:0>10}', fig)
        [model.train() for model in models]

