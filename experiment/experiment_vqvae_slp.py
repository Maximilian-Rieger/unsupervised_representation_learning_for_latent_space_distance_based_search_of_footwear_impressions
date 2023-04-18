import math
import os
import asyncio
import datetime
from typing import Tuple, List

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

from models.utils import GMSDLoss
from models.vqvae.vqvae import VQVAE, VQVAE_SLP
from optimizers.Ranger_Adabelief import RangerAdaBelief
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
                'in_chan': 1,
                'h_dim': 128,
                'res_h_dim': 64,
                'n_res_layers': 4,
                'n_embeddings': 512,
                'embedding_dim': 128,
                'beta': 0.25,
                'save_img_embedding_map': False,
                'batch_norm': False
            },

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
    def model(
            batch_size,
            in_chan,
            h_dim,
            res_h_dim,
            n_res_layers,
            n_embeddings,
            embedding_dim,
            beta,
            save_img_embedding_map,
            batch_norm
    ) -> [[nn.Module], [nn.Module]]:
        vqvae = VQVAE_SLP(
            in_chan,
            h_dim,
            res_h_dim,
            n_res_layers,
            n_embeddings,
            embedding_dim,
            beta,
            save_img_embedding_map,
            batch_norm
        ).to(GPU.device)

        if batch_size is not None:
            summary(vqvae.encoder, input_size=(vqvae.encoder.in_dim, 256, 256), batch_size=batch_size)
            summary(vqvae.decoder, input_size=(vqvae.decoder.in_dim, 1, 1), batch_size=batch_size)

        recon_loss = torch.nn.MSELoss().to(GPU.device)
        gmsd_loss = GMSDLoss(in_chan).to(GPU.device)

        return [[vqvae], [recon_loss, gmsd_loss]]

    @staticmethod
    def load_model(
            path,
            in_chan,
            h_dim,
            res_h_dim,
            n_res_layers,
            n_embeddings,
            embedding_dim,
            beta,
            save_img_embedding_map,
            batch_norm
    ):
        models = Experiment.model(
            None,
            in_chan=in_chan,
            h_dim=h_dim,
            res_h_dim=res_h_dim,
            n_res_layers=n_res_layers,
            n_embeddings=n_embeddings,
            embedding_dim=embedding_dim,
            beta=beta,
            save_img_embedding_map=save_img_embedding_map,
            batch_norm=batch_norm
        )[0]

        models, _, args, _ = helper.load_checkpoint(models, None, args={'resume': path})
        return models

    @staticmethod
    def optimizer(models, lr, step_size, beta1=0.9, beta2=0.99, gamma=0.1):
        vqvae, = models

        optimizer_vqvae = AdaBelief(vqvae.parameters(), lr=lr, betas=(beta1, beta2))

        scheduler_vqvae = lr_scheduler.StepLR(optimizer_vqvae, step_size=step_size, gamma=gamma)

        return [[optimizer_vqvae], [scheduler_vqvae]]

    def setup(self) -> SetupStruct:
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            # transforms.RandomResizedCrop(img_shape[1:], (0.8, 1.0)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

        # initialize training data
        self.training = Training(**self.args['training'], transform=self.transform, logger=self.logger)
        # self.validate = Validate(**self.args['validation'], transform=self.transform, logger=self.logger)

        models, losses = Experiment.model(self.args['batchsize'], **self.args['model'])
        optimizers, schedulers = Experiment.optimizer(models, **self.args['optimizer'])

        # torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = False

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

    def train(self, models, losses, optimizers, imgs):
        # Configure x
        real_imgs = Variable(imgs.type(Tensor).to(GPU.device))

        vqvae, = models
        mse_loss, gmsd_loss = losses
        optimizer_vqvae, = optimizers

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_vqvae.zero_grad()

        embedding_loss, x_hat = vqvae(real_imgs)[0:2]
        embedding_loss = embedding_loss.mean()
        recon_loss = mse_loss(real_imgs, x_hat)
        extra_recon_loss = gmsd_loss(x_hat, real_imgs)

        self.logger.log_value_and_epoch_avg('embedding_loss_train', embedding_loss, self.epoch, self.dataloader.batch_size)
        self.logger.log_value_and_epoch_avg('recon_loss_train', recon_loss, self.epoch, self.dataloader.batch_size)
        self.logger.log_value_and_epoch_avg('gmsd_loss_train', extra_recon_loss, self.epoch, self.dataloader.batch_size)

        loss = embedding_loss + recon_loss + extra_recon_loss

        loss.backward()

        optimizer_vqvae.step()

        return loss

    def __call__(self, setup: SetupStruct, epoch: int, menu):
        models, losses, optimizers, schedulers = setup
        [model.train() for model in models]
        scheduler_vqvae, = schedulers

        self.epoch = epoch
        self.grad_vis_g_done = not self.grad_vis
        self.grad_vis_d_done = not self.grad_vis

        self.logger.log_value('lr_vqvae', scheduler_vqvae.get_last_lr()[0], Training.global_step)

        running_loss = 0.0
        # running_acc = 0.0

        pbar = tqdm(self.dataloader)
        pbar.set_description('Epoch {}'.format(epoch))

        for i, imgs in enumerate(pbar):
            batches_done = epoch * len(self.data) + i
            loss = self.train(models, losses, optimizers, imgs)

            if self.sample_interval is not None and batches_done % self.sample_interval == 0:
                asyncio.run(self.sample_image(models, n_row=self.dataloader.batch_size, batches_done=batches_done, epoch=epoch))

            # log values
            self.logger.step(self.dataloader.batch_size)
            Training.global_step += self.dataloader.batch_size

            # calculate log values
            running_loss += loss.item() * self.dataloader.batch_size
            # running_acc += accuracy * self.dataloader.batch_size

            self.logger.log_value_and_epoch_avg('train_loss', loss.item(), self.epoch, self.dataloader.batch_size)
            # self.logger.log_value('train_accuracy', accuracy, Training.global_step)

            # menu()

        # calculate log values
        epoch_loss = running_loss / len(self.data)
        # epoch_acc = running_acc / len(self.data)

        self.logger.log_value('train_epoch_loss', epoch_loss, Training.global_step)

        # self.logger.log_value('train_epoch_accuracy', epoch_acc, Training.global_step)
        logging.info('Training {} epoch {} Lr: {:.6f} Loss: {:.6f}'.format('Impress_VQ-VAE', epoch, scheduler_vqvae.get_last_lr()[0], epoch_loss))

        if not self.grad_vis_g_done:
            helper.plot_grad_flow_lines(models[0].named_parameters(),
                                        "{}/data/Grad_flow_vqvae_{}.lines.png".format(self.logger.log_dir, self.epoch))
            self.grad_vis_g_done = True

        scheduler_vqvae.step()

        # do checkpointing
        if self.n_checkpoints is None or epoch % self.n_checkpoints == 0:
            helper.save_checkpoint(epoch, Training.global_step, models, optimizers, self.logger.log_dir,
                                   self.n_checkpoints)

        return  # do training

    def get_vector_grid(self, vqvae, n_rows=8):
        vector_stack = torch.zeros((vqvae.vector_quantization.n_e, 1, 256, 256))
        for vec in range(vqvae.vector_quantization.n_e):
            vectors = vqvae.vector_quantization.embedding.weight[vec].view((1,vqvae.vector_quantization.e_dim,1,1))
            vectors = vqvae.decoder(vectors)
            vector_stack[vec,:,:,:] = vectors.squeeze(0)
        vector_grid = make_grid(vector_stack, nrow=vqvae.vector_quantization.n_e // n_rows)
        return vector_grid

    async def sample_image(self, models, n_row, batches_done, epoch):
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

            # self.logger.log_graph(vqvae, imgs.detach())

            batch, _, _ = vqvae.reconstruct(imgs.detach())

            batch_grid = make_grid(batch.detach(), nrow=n_row // 2)
            input_grid = make_grid(imgs, nrow=n_row // 2)

            batch_stack = torch.stack([input_grid, batch_grid])
            img_grid = make_grid(batch_stack, nrow=1)
            img_grid = helper.normalize(img_grid)
            save_image(img_grid.data, "{}/data/Reconstruction_{}_{}.png".format(self.logger.log_dir, epoch, batches_done), nrow=n_row)
            # helper.save_image(img_grid, "{}/data/Reconstruction_{}.pil".format(self.logger.log_dir, batches_done))
            fig, subplots = helper.create_image_figure(img_grid.cpu(), 'Reconstructed')
            self.logger.add_figure('{}_Reconstruction_{}_{:0>10}'.format('Impress_VQ-VAE', epoch, batches_done), fig)
            # fig.savefig("{}/data/Reconstruction_fig_{:0>10}.png".format(self.logger.log_dir, batches_done))

            vector_grid = self.get_vector_grid(vqvae)
            vector_grid = helper.normalize(vector_grid)
            save_image(vector_grid.data, "{}/data/vector_vis_new_{}_{}.png".format(self.logger.log_dir, epoch, batches_done), nrow=n_row)
            fig, subplots = helper.create_image_figure(vector_grid.cpu(), 'Vector_visualization')
            self.logger.add_figure('{}_Vector_visualization_{}_new_{:0>10}'.format('Impress_VQ-VAE', epoch, batches_done), fig)
        [model.train() for model in models]

