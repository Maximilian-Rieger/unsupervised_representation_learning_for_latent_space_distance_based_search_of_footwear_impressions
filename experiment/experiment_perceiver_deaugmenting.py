import os
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
        self.args = {
            'log_dir': None,
            'config': None,
            'resume': None,
            'no_cuda': False,
            'cuda': True,
            'gpuid': 0,
            'name': None,

            'start_epoch': 0,
            'pre_training_epochs': 1,
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

            'pre_training': {
                'data': None,

                'batchsize': 1,
                'shuffle': True,
                'worker': 0,
                'sample_interval': None,
                'n_checkpoints': None,
                'grad_vis': False
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
        self.args = helper.forward_arguments(self.args, ['batchsize'], ['pre_training', 'training', 'validation'])

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
        # recon_loss = torch.nn.SmoothL1Loss().to(GPU.device)
        return [[perceiver, deceiver], [recon_loss]]

    @staticmethod
    def load_model(path, *args):
        models = Experiment.model(*args)[0]
        models, _, args, _ = helper.load_checkpoint(models, None, args={'resume': path})
        return models

    @staticmethod
    def optimizer(models, lr, step_size, beta1=0.9, beta2=0.99, gamma=0.1):
        perceiver, deceiver = models

        # optimizer = torch.optim.AdamW(itertools.chain(perceiver.parameters(), deceiver.parameters()), lr=lr, betas=(beta1, beta2))
        optimizer = AdaBelief(itertools.chain(perceiver.parameters(), deceiver.parameters()), lr=lr, betas=(beta1, beta2))

        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        return [[optimizer], [scheduler]]

    def setup(self) -> SetupStruct:
        jitter_range = (0.05, 0.95)
        rotation_range = 13  # (-30, 30)
        rotation_range = (-30, 30)
        pre_crop_range = (0.95, 1)
        self.pre_transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ])
        self.transform = DualTransforms(
            shared_transforms=[
                transforms.Resize((512, 512)),
                transforms.Grayscale(),
            ],
            transforms=[
                transforms.RandomOrder([
                    transforms.RandomResizedCrop((128, 128), pre_crop_range),
                    transforms.RandomRotation(rotation_range, fill=255),
                ]),
                transforms.ColorJitter(contrast=jitter_range),
            ],
            shared_transforms_post=[
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ],
        )
        self.validation_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

        # initialize training data
        self.pre_training = PreTraining(**self.args['pre_training'], transform=self.pre_transform, logger=self.logger)
        self.training = Training(**self.args['training'], transform=self.transform, logger=self.logger)
        self.validate = Validation(**self.args['validation'], transform=self.validation_transform, logger=self.logger)

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
        pre_end = start + self.args['pre_training_epochs']
        end = start + self.args['epochs']
        Path(os.path.join(self.logger.log_dir, 'started_trainings_loop.txt')).touch()

        for epoch in range(start, pre_end + 1):
            self.logger.epoch = epoch
            self.pre_training(setup=setup, epoch=epoch, menu=self.keyboard_menu)
        for epoch in range(start, end + 1):
            self.logger.epoch = epoch
            self.training(setup=setup, epoch=epoch, menu=self.keyboard_menu)
            self.validate(setup=setup, epoch=epoch, menu=self.keyboard_menu)

        helper.save_checkpoint(end + 1, Training.global_step, setup[0], setup[2], self.logger.log_dir, self.args['training']['n_checkpoints'])

class PreTraining:
    global_step = 0

    def __init__(self, data, batchsize, shuffle, worker, sample_interval, n_checkpoints, logger, grad_vis, **kwargs):
        self.data = DataZoo.get(**data, **kwargs)
        self.data_len = len(self.data)
        self.batch_size = batchsize
        self.batch_count = self.data_len // self.batch_size
        self.dataloader = DataLoader(self.data, batch_size=batchsize, shuffle=shuffle, num_workers=worker)

        self.name = 'Impress_Perceiver_random_masked'
        self.logger = logger
        self.sample_interval = sample_interval
        self.n_checkpoints = n_checkpoints
        self.epoch = None
        self.grad_vis = grad_vis
        self.grad_vis_done = not grad_vis
        self.prev_running_loss = 0.0

    def train(self, models, losses, optimizers, imgs):
        # Configure x
        real_imgs = Variable(imgs.type(Tensor).to(GPU.device))

        input_imgs = rearrange(real_imgs, 'b c h w -> b h w c')
        img_shape = input_imgs.shape

        perceiver, deceiver = models
        mse_loss, = losses
        optimizer, = optimizers

        # -----------------
        #  Train Generator
        # -----------------

        optimizer.zero_grad()
        mask = torch.tensor(np.random.normal(0,1,img_shape)>0.25).to(GPU.device)
        latents = perceiver(input_imgs, mask=mask)

        x_hat = deceiver(latents, out_shape=img_shape)
        recon_loss = mse_loss(input_imgs, x_hat)

        self.logger.log_value('recon_loss_train', recon_loss.item())

        loss = recon_loss
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
            imgs = torch.stack(imgs)
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



class Training:
    global_step = 0

    def __init__(self, data, batchsize, shuffle, worker, sample_interval, n_checkpoints, logger, grad_vis, **kwargs):
        self.data = DataZoo.get(**data, **kwargs)
        self.data_len = len(self.data)
        self.batch_size = batchsize
        self.batch_count = self.data_len // self.batch_size
        self.dataloader = DataLoader(self.data, batch_size=batchsize, shuffle=shuffle, num_workers=worker)

        self.name = 'Impress_Perceiver_deaugment'
        self.logger = logger
        self.sample_interval = sample_interval
        self.n_checkpoints = n_checkpoints
        self.epoch = None
        self.grad_vis = grad_vis
        self.grad_vis_done = not grad_vis
        self.prev_running_loss = 0.0

    def train(self, models, losses, optimizers, imgs):
        # Configure x
        original, augmented = imgs
        augmented = Variable(augmented.type(Tensor).to(GPU.device))
        original = Variable(original.type(Tensor).to(GPU.device))

        input_imgs = rearrange(augmented, 'b c h w -> b h w c')
        target_imgs = rearrange(original, 'b c h w -> b h w c')
        img_shape = input_imgs.shape

        perceiver, deceiver = models
        mse_loss, = losses
        optimizer, = optimizers

        # -----------------
        #  Train Generator
        # -----------------

        optimizer.zero_grad()
        latents = perceiver(input_imgs)

        x_hat = deceiver(latents, out_shape=img_shape)
        recon_loss = mse_loss(x_hat, target_imgs)

        self.logger.log_value('recon_loss_train', recon_loss.item())

        loss = recon_loss
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

            res = self.data[batch_start:batch_end]
            augmented, original = [], []
            for imgs in res:
                original += [imgs[0]]
                augmented += [imgs[1]]
            augmented_imgs = torch.stack(augmented)
            original_imgs = torch.stack(original)
            augmented_imgs = augmented_imgs.to(GPU.device)
            original_imgs = original_imgs.to(GPU.device)

            batch = rearrange(augmented_imgs, 'b c h w -> b h w c')
            img_shape = batch.shape

            batch = deceiver(perceiver(batch), img_shape, mask=None)
            batch = rearrange(batch, 'b h w c -> b c h w')

            batch_grid = make_grid(batch.detach(), nrow=n_row // 2)
            input_grid = make_grid(augmented_imgs, nrow=n_row // 2)
            target_grid = make_grid(original_imgs, nrow=n_row // 2)

            batch_stack = torch.stack([input_grid, target_grid, batch_grid])
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
        mse_loss, = losses

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

