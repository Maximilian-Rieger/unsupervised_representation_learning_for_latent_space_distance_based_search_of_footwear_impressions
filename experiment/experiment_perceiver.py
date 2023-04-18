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
from models.experimental.perceiver_2 import Perceiver as Perceiver, Deceiver as Deceiver
from dataloading.transforms import DualTransforms

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
        self.validation = None
        self.training = None
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
            'img_size': (256, 256),

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

            'use_validation': True,
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

        self.logger = TensorboardLogger(os.path.join(args['log_dir'], args['name']), modules=[__name__], images_dir=True)

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
            **kvargs,
    ) -> [[nn.Module], [nn.Module]]:
        encoderArgs = {
            **kvargs,
            'depth': kvargs['encoder_depth'],
        }
        del encoderArgs['encoder_depth']
        del encoderArgs['decoder_depth']
        decoderArgs = {
            **kvargs,
            'depth': kvargs['decoder_depth'],
        }
        del decoderArgs['encoder_depth']
        del decoderArgs['decoder_depth']

        perceiver = Perceiver(**encoderArgs).to(GPU.device)
        deceiver = Deceiver(**decoderArgs).to(GPU.device)

        summary(perceiver, input_size=(128, 128, 1), batch_size=batch_size)
        summary(FuncApplyWrapper(deceiver, lambda _: [[2, 128, 128, 1]]), input_size=(kvargs['latent_dim'],), batch_size=batch_size)
        # summary(perceiver, input_size=(128, 128, 3), batch_size=batch_size)
        # summary(FuncApplyWrapper(deceiver, lambda _: [[2, 128, 128, 1]]), input_size=(latent_dim,), batch_size=batch_size)

        recon_loss = torch.nn.MSELoss().to(GPU.device)
        cosine_sim_loss = torch.nn.CosineSimilarity()

        return [[perceiver, deceiver], [recon_loss, cosine_sim_loss]]

    @staticmethod
    def load_model(path, **kvargs):
        models = Experiment.model(**kvargs)[0]
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
        # initialize training data

        self.training = Training(
            **self.args['training'],
            transform=DualTransforms(
                shared_transforms=transforms.Compose([
                    transforms.Resize(self.args['img_size']),
                    # transforms.Grayscale(),
                ]),
                transforms=transforms.Compose([
                    transforms.RandomHorizontalFlip(p=1),
                    transforms.RandomResizedCrop(self.args['img_size'], scale=(0.8, 1.0)),
                ]),
                shared_transforms_post=transforms.ToTensor(),
            ),
            logger=self.logger
        )
        if self.args['use_validation']:
            self.validation = Validation(
                **self.args['validation'],
                transform=transforms.Compose([
                    transforms.Resize(self.args['img_size']),
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                ]),
                logger=self.logger
            )
        else:
            self.validation = lambda *_, **__: None

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

        for epoch in range(0, end + 1):
            self.logger.epoch = epoch
            self.training(setup=setup, epoch=self.logger.epoch, menu=self.keyboard_menu)
            self.validation(setup=setup, epoch=self.logger.epoch, menu=self.keyboard_menu)

        helper.save_checkpoint(end + 1, Training.global_step, setup[0], setup[2], self.logger.log_dir, self.args['training']['n_checkpoints'])


class Training:
    global_step = 0

    def __init__(self, data, batchsize, shuffle, worker, sample_interval, n_checkpoints, logger, grad_vis, **kwargs):
        self.data = DataZoo.get(**data, **kwargs)
        self.data_len = len(self.data)
        self.batch_size = batchsize
        self.batch_count = self.data_len // self.batch_size
        self.dataloader = DataLoader(self.data, batch_size=batchsize, shuffle=shuffle, num_workers=worker)

        self.name = 'Impress_Perceiver'
        self.logger = logger
        self.sample_interval = sample_interval
        self.n_checkpoints = n_checkpoints
        self.epoch = None
        self.grad_vis = grad_vis
        self.grad_vis_done = not grad_vis
        self.prev_running_loss = 0.0

    def train(self, models, losses, optimizers, imgs):
        # Configure x
        imgs_a, imgs_b = imgs
        real_imgs_a = Variable(imgs_a.type(Tensor).to(GPU.device))
        real_imgs_b = Variable(imgs_b.type(Tensor).to(GPU.device))

        input_imgs_a = rearrange(real_imgs_a, 'b c h w -> b h w c')
        input_imgs_b = rearrange(real_imgs_b, 'b c h w -> b h w c')
        img_shape = input_imgs_a.shape

        perceiver, deceiver = models
        mse_loss, cosine_sim_loss = losses
        optimizer, = optimizers

        # ------------------
        #  Train Similarity
        # ------------------

        optimizer.zero_grad()
        latents_a = perceiver(input_imgs_a)
        latents_b = perceiver(input_imgs_b)
        sim_loss = (1 - cosine_sim_loss(latents_a, latents_b)).mean()
        self.logger.log_value('sim_loss_train', sim_loss.item())

        # ----------------------
        #  Train Reconstruction
        # ----------------------

        x_hat = deceiver(latents_a, out_shape=img_shape)
        recon_loss = mse_loss(input_imgs_a, x_hat)

        x_hat = deceiver(latents_b, out_shape=img_shape)
        recon_loss += mse_loss(input_imgs_b, x_hat)
        recon_loss /= 2

        self.logger.log_value('recon_loss_train', recon_loss.item())

        loss = recon_loss + sim_loss
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

        for i, imgs in enumerate(pbar):
            batches_done = epoch * self.batch_count + i
            loss = self.train(models, losses, optimizers, imgs)

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
            imgs = [item for sublist in zip([a for a, _ in imgs], [b for _, b in imgs]) for item in sublist]
            imgs = torch.stack(imgs, dim=0)
            imgs = imgs.to(GPU.device)

            batch = rearrange(imgs, 'b c h w -> b h w c')
            img_shape = batch.shape

            batch = deceiver(perceiver(batch), img_shape, mask=None)
            batch = rearrange(batch, 'b h w c -> b c h w')

            batch_grid = make_grid(batch.detach(), nrow=n_row // 2)
            input_grid = make_grid(imgs, nrow=n_row // 2)

            batch_stack = torch.stack([input_grid, batch_grid])
            img_grid = make_grid(batch_stack, nrow=1)
            img_grid = helper.normalize(img_grid)
            save_image(img_grid.data, "{}/data/Reconstruction_{}_{}.png".format(self.logger.log_dir, self.epoch, batches_done), nrow=n_row)
            fig, subplots = helper.create_image_figure(img_grid.cpu(), 'Reconstructed')

            self.logger.add_figure('{}_Reconstruction_{}_{:0>10}'.format(self.name, self.epoch, batches_done), fig)
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

