import math
import os
import asyncio
import datetime
from typing import Tuple, List

import logging
from pathlib import Path
import faiss
import torch
from torch import nn, Tensor
import torch.nn.init
import torch.utils.data
from torch.optim import lr_scheduler
from torch.autograd import Variable
import matplotlib as mpl
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
import itertools
import numpy as np
from models.experimental.perceiver_2 import PerceiverLT as Perceiver, DeceiverLT as Deceiver
from optimizers.AdaBelief import AdaBelief
from dataloading.transforms import NTransforms

from einops import rearrange

mpl.use('Agg')

SetupStruct = Tuple[
    List[nn.Module],
    List[nn.Module],
    List[torch.optim.Optimizer],
    List[torch.optim.lr_scheduler._LRScheduler]
]


class ApplyWrapper(nn.Module):
    def __init__(self, wrapped, forward_pos=1) -> None:
        super().__init__()
        self.wrapped = wrapped
        self.forward_pos = forward_pos

    def forward(self, x):
        return self.wrapped(x)[self.forward_pos]


class FuncApplyWrapper(nn.Module):
    def __init__(self, wrapped, func) -> None:
        super().__init__()
        self.wrapped = wrapped
        self.func = func

    def forward(self, x):
        return self.wrapped(x, *self.func(x))


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

            'model': {
                'num_freq_bands': 6,
                'encoder_depth': 3,
                'decoder_depth': 3,
                'max_freq': 64,
                'freq_base': 2,
                'input_channels': 1,
                'input_axis': 2,
                'num_latents': 512,
                'latent_dim': 512,
                'cross_heads': 1,
                'latent_heads': 8,
                'cross_dim_head': 64,
                'latent_dim_head': 64,
                'attn_dropout': 0.,
                'ff_dropout': 0.,
                'weight_tie_layers': False
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
                'save_best_model': True,
                'batchsize': 1,
                'worker': 0,
                'sample_interval': None,
            }
        }

        self.keyboard_menu = None
        self.image_transform = None
        self.label_transform = None

        # check config by comparing with defaults and merge
        changes = dict_merge(self.args, args, verify=True)

        # forward arguments
        self.args = helper.forward_arguments(self.args, ['batchsize'], ['training', 'validation'])

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
            input_channels,
            num_freq_bands,
            encoder_depth,
            decoder_depth,
            max_freq,
            freq_base,
            input_axis,
            num_latents,
            latent_dim,
            cross_heads,
            latent_heads,
            cross_dim_head,
            latent_dim_head,
            attn_dropout,
            ff_dropout,
            weight_tie_layers,
    ) -> [[nn.Module], [nn.Module]]:
        perceiver = Perceiver(
            depth=encoder_depth,
            num_freq_bands=num_freq_bands,
            max_freq=max_freq,
            input_channels=input_channels,
            freq_base=freq_base,
            input_axis=input_axis,
            num_latents=num_latents,
            latent_dim=latent_dim,
            cross_heads=cross_heads,
            latent_heads=latent_heads,
            cross_dim_head=cross_dim_head,
            latent_dim_head=latent_dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            weight_tie_layers=weight_tie_layers,
        ).to(GPU.device)
        deceiver = Deceiver(
            depth=decoder_depth,
            num_freq_bands=num_freq_bands,
            max_freq=max_freq,
            input_channels=input_channels,
            freq_base=freq_base,
            input_axis=input_axis,
            num_latents=num_latents,
            latent_dim=latent_dim,
            cross_heads=cross_heads,
            latent_heads=latent_heads,
            cross_dim_head=cross_dim_head,
            latent_dim_head=latent_dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            weight_tie_layers=weight_tie_layers,
        ).to(GPU.device)

        summary(perceiver, input_size=(128, 128, 1), batch_size=batch_size)
        summary(FuncApplyWrapper(deceiver, lambda _: [[2, 128, 128, 1]]), input_size=(latent_dim,), batch_size=batch_size)

        recon_loss = torch.nn.MSELoss().to(GPU.device)
        cosine_sim_loss = torch.nn.CosineSimilarity(dim=-1).to(GPU.device)

        return [[perceiver, deceiver], [recon_loss, cosine_sim_loss]]

    @staticmethod
    def load_model(path, *args):
        models = Experiment.model(*args)[0]
        models, _, args, _ = helper.load_checkpoint(models, None, args={'resume': path})
        return models

    @staticmethod
    def optimizer(models, lr, step_size, beta1=0.9, beta2=0.99, gamma=0.1):
        perceiver, deceiver = models

        optimizer = torch.optim.AdamW(itertools.chain(perceiver.parameters(), deceiver.parameters()), lr=lr, betas=(beta1, beta2))
        # optimizer = AdaBelief(itertools.chain(perceiver.parameters(), deceiver.parameters()), lr=lr, betas=(beta1, beta2))

        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        return [[optimizer], [scheduler]]

    def setup(self) -> SetupStruct:
        self.transforms = NTransforms(
            # 4,
            2,
            transforms=[
                # [transforms.Resize((16, 16))],
                [transforms.Resize((32, 32))],
                # [transforms.Resize((64, 64))],
                [transforms.Resize((128, 128))],
            ],
            shared_transforms_post=[
                transforms.Grayscale(),
                transforms.ToTensor()
            ]
        )
        self.transform_val = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

        # initialize training data
        self.training = Training(**self.args['training'], transform=self.transforms, logger=self.logger)
        self.validation = Validation(**self.args['validation'], transform=self.transform_val, logger=self.logger)

        models, losses = Experiment.model(self.args['batchsize'], **self.args['model'])
        optimizers, schedulers = Experiment.optimizer(models, **self.args['optimizer'])

        if self.args['resume'] is not None:
            models, optimizers, self.args, train_step = helper.load_checkpoint(models, optimizers, self.args)
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
            self.validation(setup=setup, epoch=epoch, menu=self.keyboard_menu)

        helper.save_checkpoint(end + 1, Training.global_step, setup[0], setup[2], self.logger.log_dir, self.args['training']['n_checkpoints'])


class Training:
    global_step = 0

    def __init__(self, data, batchsize, shuffle, worker, sample_interval, n_checkpoints, logger, grad_vis, **kwargs):
        self.data = DataZoo.get(**data, **kwargs)
        self.data_len = len(self.data)
        self.batch_size = batchsize
        self.batch_count = self.data_len // self.batch_size
        self.dataloader = DataLoader(self.data, batch_size=batchsize, shuffle=shuffle, num_workers=worker)

        self.name = 'Impress_Perceiver_steps'
        self.logger = logger
        self.sample_interval = sample_interval
        self.n_checkpoints = n_checkpoints
        self.epoch = None
        self.grad_vis = grad_vis
        self.grad_vis_done = not grad_vis
        self.prev_running_loss = 0.0

    def update_dataloader(self, data, batchsize, shuffle, worker, **kwargs):
        self.data = DataZoo.get(**data, **kwargs)
        self.data_len = len(self.data)
        self.batch_count = self.data_len // self.batch_size
        self.dataloader = DataLoader(self.data, batch_size=batchsize, shuffle=shuffle, num_workers=worker)

    def train(self, models, losses, optimizers, data):
        # initialize model
        perceiver, deceiver = models
        mse_loss, cosine_sim_loss = losses
        optimizer, = optimizers

        # Configure x
        # imgs_16, imgs_32, imgs_64, imgs_128 = data
        imgs_32, imgs_128 = data
        # real_imgs_16 = Variable(imgs_16.type(Tensor).to(GPU.device))
        real_imgs_32 = Variable(imgs_32.type(Tensor).to(GPU.device))
        # real_imgs_64 = Variable(imgs_64.type(Tensor).to(GPU.device))
        real_imgs_128 = Variable(imgs_128.type(Tensor).to(GPU.device))

        # input_imgs_16 = rearrange(real_imgs_16, 'b c h w -> b h w c')
        input_imgs_32 = rearrange(real_imgs_32, 'b c h w -> b h w c')
        # input_imgs_64 = rearrange(real_imgs_64, 'b c h w -> b h w c')
        input_imgs_128 = rearrange(real_imgs_128, 'b c h w -> b h w c')
        del real_imgs_32, real_imgs_128

        # img_shape_16 = input_imgs_16.shape
        img_shape_32 = input_imgs_32.shape
        # img_shape_64 = input_imgs_64.shape
        img_shape_128 = input_imgs_128.shape

        # -----------------
        #  Train Generator
        # -----------------

        optimizer.zero_grad()
        # latents_16 = perceiver(input_imgs_16)
        latents_32 = perceiver(input_imgs_32)
        # latents_64 = perceiver(input_imgs_64)
        latents_128 = perceiver(input_imgs_128)

        # sim_loss_16_32 = cosine_sim_loss(latents_16, latents_32)
        # self.logger.log_value('cosine_sim_loss_train_16_32', sim_loss_16_32.mean())
        # sim_loss_64_128 = cosine_sim_loss(latents_64, latents_128)
        # self.logger.log_value('cosine_sim_loss_train_64_128', sim_loss_64_128.mean())
        # sim_loss_32_64 = cosine_sim_loss(latents_32, latents_128)
        # self.logger.log_value('cosine_sim_loss_train_32_64', sim_loss_32_64.mean())
        sim_loss_32_128 = 1 - cosine_sim_loss(latents_32, latents_128)
        # self.logger.log_value('cosine_sim_loss_train_sim_loss_32_128', sim_loss_32_128.mean())
        # sim_loss = sim_loss_16_32 + sim_loss_32_64 + sim_loss_64_128
        # del sim_loss_16_32, sim_loss_32_64, sim_loss_64_128
        sim_loss = sim_loss_32_128.mean()
        self.logger.log_value('cosine_sim_loss_train_full', sim_loss.item())
        #
        # recon_loss_16_16 = mse_loss(deceiver(latents_16, out_shape=img_shape_16), input_imgs_16)
        # recon_loss_32_16 = mse_loss(deceiver(latents_32, out_shape=img_shape_16), input_imgs_16)
        # recon_loss_64_16 = mse_loss(deceiver(latents_64, out_shape=img_shape_16), input_imgs_16)
        # recon_loss_128_16 = mse_loss(deceiver(latents_128, out_shape=img_shape_16), input_imgs_16)
        # recon_loss_16 = recon_loss_16_16 + recon_loss_32_16 + recon_loss_64_16 + recon_loss_128_16
        # self.logger.log_value('recon_loss_train_16', recon_loss_16.item())

        # recon_loss_16_32 = mse_loss(deceiver(latents_16, out_shape=img_shape_32), input_imgs_32)
        recon_loss_32_32 = mse_loss(deceiver(latents_32, out_shape=img_shape_32), input_imgs_32)
        self.logger.log_value('recon_loss_train_32_32', recon_loss_32_32.item())
        # recon_loss_64_32 = mse_loss(deceiver(latents_64, out_shape=img_shape_32), input_imgs_32)
        recon_loss_128_32 = mse_loss(deceiver(latents_128, out_shape=img_shape_32), input_imgs_32)
        self.logger.log_value('recon_loss_train_128_32', recon_loss_128_32.item())
        # recon_loss_32 = mse_loss(x_hat_16_32, input_imgs_32) + mse_loss(x_hat_32_32, input_imgs_32) + mse_loss(x_hat_64_32, input_imgs_32) + mse_loss(x_hat_128_32, input_imgs_32)
        recon_loss_32 = recon_loss_32_32 + recon_loss_128_32
        del recon_loss_32_32, recon_loss_128_32
        self.logger.log_value('recon_loss_train_32', recon_loss_32.item())

        # x_hat_16_64 = deceiver(latents_16, out_shape=img_shape_64)
        # x_hat_32_64 = deceiver(latents_32, out_shape=img_shape_64)
        # x_hat_64_64 = deceiver(latents_64, out_shape=img_shape_64)
        # x_hat_128_64 = deceiver(latents_128, out_shape=img_shape_64)
        # recon_loss_64 = mse_loss(x_hat_16_64, input_imgs_64) + mse_loss(x_hat_32_64, input_imgs_64) + mse_loss(x_hat_64_64, input_imgs_64) + mse_loss(x_hat_128_64, input_imgs_64)
        # self.logger.log_value('recon_loss_train_64', recon_loss_64.item())

        # recon_loss_16_128 = mse_loss(deceiver(latents_16, out_shape=img_shape_128), input_imgs_128)
        recon_loss_32_128 = mse_loss(deceiver(latents_32, out_shape=img_shape_128), input_imgs_128)
        self.logger.log_value('recon_loss_train_32_128', recon_loss_32_128.item())
        # recon_loss_64_128 = mse_loss(deceiver(latents_64, out_shape=img_shape_128), input_imgs_128)
        recon_loss_128_128 = mse_loss(deceiver(latents_128, out_shape=img_shape_128), input_imgs_128)
        self.logger.log_value('recon_loss_train_128_128', recon_loss_128_128.item())
        # recon_loss_128 = mse_loss(x_hat_16_128, input_imgs_128) + mse_loss(x_hat_32_128, input_imgs_128) + mse_loss(x_hat_64_128, input_imgs_128) + mse_loss(x_hat_128_128, input_imgs_128)
        recon_loss_128 = recon_loss_32_128 + recon_loss_128_128
        self.logger.log_value('recon_loss_train_128', recon_loss_128.item())

        # recon_loss = recon_loss_16 + recon_loss_32 + recon_loss_64 + recon_loss_128
        recon_loss = recon_loss_32 + recon_loss_128
        self.logger.log_value('recon_loss_train_full', recon_loss.item())

        loss = recon_loss + sim_loss
        # loss = recon_loss
        loss.backward()

        optimizer.step()

        return loss

    def __call__(self, setup: SetupStruct, epoch: int, menu):
        models, losses, optimizers, schedulers = setup
        [model.train() for model in models]
        scheduler, = schedulers

        self.epoch = epoch
        self.grad_vis_done = not self.grad_vis

        self.logger.log_value('lr', scheduler.get_last_lr()[0], Training.global_step)

        running_loss = 0.0

        pbar = tqdm(self.dataloader)
        pbar.set_description('Epoch {}'.format(epoch))

        scaled_sample_interval = self.sample_interval * self.batch_count

        for i, data in enumerate(pbar):
            batches_done = epoch * self.batch_count + i
            loss = self.train(models, losses, optimizers, data)

            if self.sample_interval is not None and batches_done % scaled_sample_interval == 0:
                self.sample_image(models, n_row=self.batch_size, batches_done=batches_done)

            # log values
            self.logger.step(self.batch_size)
            Training.global_step += self.batch_size

            # calculate log values
            running_loss += loss.item() * self.batch_size

            self.logger.log_value_and_epoch_avg('train_loss', loss.item(), self.epoch, self.batch_size, self.data_len)

        # calculate log values
        epoch_loss = running_loss / self.data_len
        self.prev_running_loss = running_loss
        self.logger.log_value('train_epoch_loss', epoch_loss, Training.global_step)

        logging.info('Training {} epoch {} Lr: {:.6f} Loss: {:.6f}'.format(self.name, epoch, scheduler.get_last_lr()[0], epoch_loss))

        if not self.grad_vis_done and epoch % 2 == 0:
            model_params_dict = {
                'Perceiver': models[0].named_parameters(),
                'Deceiver': models[1].named_parameters(),
            }
            helper.plot_multiple_grad_flow_lines(model_params_dict, "{}/data/Grad_flow_{}.lines.png".format(self.logger.log_dir, self.epoch))
            self.grad_vis_done = True

        scheduler.step()

        # do checkpointing
        if self.n_checkpoints is None or epoch % self.n_checkpoints == 0:
            helper.save_checkpoint(epoch, Training.global_step, models, optimizers, self.logger.log_dir, self.n_checkpoints or self.epoch)

        return  # do training

    def sample_image(self, models, n_row, batches_done):
        """Saves a grid of generated data"""
        [model.eval() for model in models]
        with torch.no_grad():
            perceiver, deceiver = models

            batch_start = ((batches_done * self.batch_size) % self.data_len)
            batch_end = ((batches_done * self.batch_size + self.batch_size) % self.data_len)

            imgs = self.data[batch_start:batch_end]
            imgs_32, imgs_128 = [x[0] for x in imgs], [x[1] for x in imgs]
            imgs_32, imgs_128 = torch.stack(imgs_32), torch.stack(imgs_128)
            imgs_32, imgs_128 = imgs_32.to(GPU.device), imgs_128.to(GPU.device)

            batch_32, batch_128 = rearrange(imgs_32, 'b c h w -> b h w c'), rearrange(imgs_128, 'b c h w -> b h w c')
            img_shape_32, img_shape_128 = batch_32.shape, batch_128.shape

            batch_32, batch_128 = deceiver(perceiver(batch_32), img_shape_32, mask=None), deceiver(perceiver(batch_128), img_shape_128, mask=None)
            batch_32, batch_128 = rearrange(batch_32, 'b h w c -> b c h w'), rearrange(batch_128, 'b h w c -> b c h w')

            batch_grid_32, batch_grid_128 = make_grid(batch_32.detach(), nrow=n_row // 2), make_grid(batch_128.detach(), nrow=n_row // 2)
            input_grid_32, input_grid_128 = make_grid(imgs_32, nrow=n_row // 2), make_grid(imgs_128, nrow=n_row // 2)

            batch_stack_32, batch_stack_128 = torch.stack([batch_grid_32, input_grid_32]), torch.stack([batch_grid_128, input_grid_128])
            img_grid_32, img_grid_128 = make_grid(batch_stack_32, nrow=1), make_grid(batch_stack_128, nrow=1)
            img_grid_32, img_grid_128 = helper.normalize(img_grid_32), helper.normalize(img_grid_128)
            save_image(img_grid_32.data, "{}/data/Reconstruction_32_{}_{}.png".format(self.logger.log_dir, self.epoch, batches_done), nrow=n_row)
            save_image(img_grid_128.data, "{}/data/Reconstruction_128_{}_{}.png".format(self.logger.log_dir, self.epoch, batches_done), nrow=n_row)
            fig, subplots = helper.create_image_figure(img_grid_32.cpu(), 'Reconstructed_32')
            self.logger.add_figure('{}_Reconstruction_32_{}_{:0>10}'.format(self.name, self.epoch, batches_done), fig)
            fig, subplots = helper.create_image_figure(img_grid_128.cpu(), 'Reconstructed_128')
            self.logger.add_figure('{}_Reconstruction_128_{}_{:0>10}'.format(self.name, self.epoch, batches_done), fig)

        [model.train() for model in models]


class Validation:
    global_step = 0

    def __init__(self, data, batchsize, worker, logger, sample_interval, save_best_model=True, **kwargs):
        self.data = DataZoo.get(**data, **kwargs)
        self.batch_size = batchsize
        self.dataloader = DataLoader(self.data, batch_size=batchsize, shuffle=False, num_workers=worker)
        self.name = 'Impress_Perceiver'
        self.logger = logger
        self.sample_interval = sample_interval
        self.epoch = None
        self.save_best_model = save_best_model
        self.best_loss = float('inf')
        self.data_len = len(self.data)
        self.batch_count = self.data_len // self.batch_size

    def train(self, models, losses, data):
        # Configure x
        labels, imgs = data

        real_imgs = Variable(imgs.type(Tensor).to(GPU.device))

        input_imgs = rearrange(real_imgs, 'b c h w -> b h w c')
        img_shape = input_imgs.shape

        perceiver, deceiver = models
        mse_loss, _ = losses

        latents = perceiver(input_imgs)
        self.logger.accumulate_embedding_set_for_epoch(latents.cpu(), labels, images=imgs, name=self.name)
        x_hat = deceiver(latents, out_shape=img_shape)
        recon_loss = mse_loss(input_imgs, x_hat)

        self.logger.log_value('recon_loss_val', recon_loss.item())

        loss = recon_loss

        loss.backward()

        return loss

    def __call__(self, setup, epoch, menu):
        self.epoch = epoch
        models, losses, optimizers, _ = setup
        [model.eval() for model in models]

        running_loss = 0.0

        pbar = tqdm(self.dataloader)
        pbar.set_description('Validation Epoch {}'.format(epoch))
        scaled_sample_interval = self.sample_interval * self.batch_count

        for i, imgs in enumerate(pbar):
            batches_done = epoch * self.batch_count + i
            loss = self.train(models, losses, imgs)

            if self.sample_interval is not None and batches_done % scaled_sample_interval == 0:
                self.sample_image(models, n_row=self.batch_size, batches_done=batches_done)

            # log values
            self.logger.step(self.batch_size)
            Validation.global_step += self.batch_size

            # calculate log values
            running_loss += loss.item() * self.batch_size

            self.logger.log_value_and_epoch_avg('validation_loss', loss.item(), self.epoch, self.batch_size, self.data_len)

        epoch_loss = self.logger.get_epoch_loss('validation_loss', self.data_len)
        if epoch_loss < self.best_loss:
            self.best_loss = epoch_loss
            helper.save_checkpoint(epoch, Training.global_step, models, optimizers, self.logger.log_dir, best=True)
            features, labels, labels_header, images = self.logger.get_embedding_set(self.name)
            clusters = len(labels.unique())
            fmeans = faiss.Kmeans(features.shape[1], clusters)
            fmeans.train(features.detach().numpy())
            cluster_centers, cluster_ids_x = fmeans.assign(features.detach().numpy())
            kmeans_acc = 0.0
            for n in range(1, clusters + 1, 2):
                kmeans_acc += 2 if cluster_ids_x[n - 1] == cluster_ids_x[n] else 0
            kmeans_acc /= clusters + 1
            self.logger.log_value('Kmeans_acc_val', kmeans_acc)

            self.logger.log_embedding_set(self.name, step=epoch)
        else:
            self.logger.clear_embedding_set(self.name)

        logging.info('Validation {} epoch {} Loss: {:.6f}'.format(self.name, epoch, epoch_loss))

        # do checkpointing
        return  # do training

    def sample_image(self, models, n_row, batches_done):
        """Saves a grid of generated data"""
        with torch.no_grad():
            perceiver, deceiver = models

            batch_start = ((batches_done * self.batch_size) % self.data_len)
            batch_end = ((batches_done * self.batch_size + self.batch_size) % self.data_len)
            data = self.data[batch_start:batch_end]
            imgs = [img for _, img in data]
            imgs = torch.stack(imgs)
            imgs = imgs.to(GPU.device)

            batch = rearrange(imgs, 'b c h w -> b h w c')
            img_shape = batch.shape

            # self.logger.log_graph(perceiver, (torch.rand((1,256,256,1),device=GPU.device),))
            encoded_batch = perceiver(batch)

            batch = deceiver(encoded_batch, img_shape)
            batch = rearrange(batch, 'b h w c -> b c h w')

            batch_grid = make_grid(batch.detach(), nrow=n_row // 2)
            # mask = Tensor(np.random.normal(0, 1, imgs.shape)).to(GPU.device) > 0.5  # 0.5 ~ 30% | 0.25 ~ 40% | 0.0 ~ 50% |

            # imgs[0:2] = imgs.masked_fill(mask, 0)[0:2]
            # second_row_slice = slice(self.batch_size//2, self.batch_size//2+2)
            # imgs[second_row_slice] = imgs.masked_fill(mask, 0)[second_row_slice]
            input_grid = make_grid(imgs, nrow=n_row // 2)

            batch_stack = torch.stack([input_grid, batch_grid])
            img_grid = make_grid(batch_stack, nrow=1)
            img_grid = helper.normalize(img_grid)
            save_image(img_grid.data, "{}/data/Reconstruction_val_{}_{}.png".format(self.logger.log_dir, self.epoch, batches_done), nrow=n_row)
            # fig, subplots = helper.create_image_figure(img_grid.cpu(), 'Reconstructed')
            # self.logger.add_figure('{}_Reconstruction_val_{}_{:0>10}'.format(self.name, self.epoch, batches_done), fig)

            z = Tensor(np.random.normal(0, 1, (self.batch_size, perceiver.latent_dim))).to(GPU.device)
            gen_imgs = deceiver(z, out_shape=img_shape)
            gen_imgs = rearrange(gen_imgs, 'b h w c -> b c h w')

            img_grid = make_grid(gen_imgs.detach(), nrow=n_row // 4, normalize=True)
            save_image(img_grid.data, "{}/data/{}_random_sample_val_{}_{}.png".format(self.logger.log_dir, self.name, self.epoch, batches_done), nrow=n_row, normalize=True)
            # fig, subplots = helper.create_image_figure(img_grid.cpu(), 'Generated_Random_Sample_val_{}_{:0>10}'.format(self.epoch, batches_done), n_row // 4, 1)
            # self.logger.add_figure('{}_Random_Sample_val_{}_{:0>10}'.format(self.name, self.epoch, batches_done), fig)

            z = Tensor(np.random.normal(0, 1, (self.batch_size, perceiver.latent_dim))).to(GPU.device)

            gen_imgs = deceiver(encoded_batch * z, out_shape=img_shape)
            gen_imgs = rearrange(gen_imgs, 'b h w c -> b c h w')

            img_grid = make_grid(gen_imgs.detach(), nrow=n_row // 4, normalize=True)
            save_image(img_grid.data, "{}/data/{}_random_skewed_sample_val_{}_{}.png".format(self.logger.log_dir, self.name, self.epoch, batches_done), nrow=n_row, normalize=True)
            # fig, subplots = helper.create_image_figure(img_grid.cpu(), 'Generated_Random_Skewed_Sample_val_{}_{:0>10}'.format(self.epoch, batches_done), n_row // 4, 1)
            # self.logger.add_figure('{}_Random_Skewed_Sample_val_{}_{:0>10}'.format(self.name, self.epoch, batches_done), fig)
