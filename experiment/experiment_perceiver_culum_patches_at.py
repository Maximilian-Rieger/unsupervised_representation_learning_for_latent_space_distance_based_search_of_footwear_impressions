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
import itertools

from models.experimental.perceiver_2 import Perceiver, Deceiver
from optimizers.AdaBelief import AdaBelief

from einops import rearrange

from models.AAE import Discriminator2 as Discriminator

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
                'num_freq_bands': 6,
                'encoder_depth': 6,
                'decoder_depth': 3,
                'max_freq': 64,
                'freq_base': 2,
                'input_channels': 3,
                'input_axis': 2,
                'num_latents': 512,
                'cross_dim': 512,
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
            input_channels,
            num_freq_bands,
            encoder_depth,
            decoder_depth,
            max_freq,
            freq_base,
            input_axis,
            num_latents,
            cross_dim,
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
            cross_dim=cross_dim,
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
            cross_dim=cross_dim,
            latent_dim=latent_dim,
            cross_heads=cross_heads,
            latent_heads=latent_heads,
            cross_dim_head=cross_dim_head,
            latent_dim_head=latent_dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            weight_tie_layers=weight_tie_layers,
        ).to(GPU.device)

        discriminator = Discriminator(latent_size=latent_dim).to(GPU.device)

        # summary(ApplyWrapper(vqvae, 1), input_size=(vqvae.encoder.in_dim, 256, 256), batch_size=batch_size)
        summary(perceiver, input_size=(178, 218, 3), batch_size=batch_size)
        summary(deceiver, input_size=(latent_dim,), batch_size=batch_size)
        summary(discriminator, input_size=(latent_dim,), batch_size=batch_size)

        recon_loss = torch.nn.SmoothL1Loss().to(GPU.device)
        # gmsd_loss = GMSDLoss(input_channels).to(GPU.device)
        adversarial_loss = torch.nn.BCELoss().to(GPU.device)

        # return [[perceiver, deceiver, discriminator], [recon_loss, gmsd_loss, adversarial_loss]]
        return [[perceiver, deceiver, discriminator], [recon_loss, adversarial_loss]]

    @staticmethod
    def load_model(path, *args):
        models = Experiment.model(*args)[0]
        models, _, args, _ = helper.load_checkpoint(models, None, args={'resume': path})
        return models

    @staticmethod
    def optimizer(models, lr, step_size, beta1=0.9, beta2=0.99, gamma=0.1):
        perceiver, deceiver, discriminator = models

        # optimizer_perceiver = AdaBelief(perceiver.parameters(), lr=lr, betas=(beta1, beta2))
        # optimizer_deceiver = AdaBelief(deceiver.parameters(), lr=lr, betas=(beta1, beta2))
        optimizer = AdaBelief(itertools.chain(perceiver.parameters(), deceiver.parameters()), lr=lr, betas=(beta1, beta2))
        optimizer_D = AdaBelief(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
        # optimizer = MADGRAD(itertools.chain(perceiver.parameters(), deceiver.parameters()), lr=lr, momentum=beta1, weight_decay=beta2)
        # optimizer_D = MADGRAD(discriminator.parameters(),lr=lr, momentum=beta1, weight_decay=beta2)

        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        scheduler_D = lr_scheduler.StepLR(optimizer_D, step_size=step_size, gamma=gamma)

        return [[optimizer, optimizer_D], [scheduler, scheduler_D]]

    def setup(self) -> SetupStruct:
        self.transform = transforms.Compose([
            # transforms.Resize((320, 320)),
            # transforms.Resize((256, 256)),
            # transforms.RandomResizedCrop((256, 256), (0.8, 1.0)),
            # transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

        # initialize training data
        self.training = Training(**self.args['training'], transform=self.transform, logger=self.logger)
        # self.validate = Validate(**self.args['validation'], transform=self.transform, logger=self.logger)

        models, losses = Experiment.model(self.args['batchsize'], **self.args['model'])
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

        for epoch in range(start, end + 1):
            self.logger.epoch = epoch
            self.training(setup=setup, epoch=epoch, menu=self.keyboard_menu)

        helper.save_checkpoint(end + 1, Training.global_step, setup[0], setup[2], self.logger.log_dir, self.args['training']['n_checkpoints'])


class Training:
    global_step = 0

    def __init__(self, data, batchsize, shuffle, worker, sample_interval, n_checkpoints, logger, grad_vis, **kwargs):
        self.data = DataZoo.get(**data, **kwargs)
        self.dataloader = DataLoader(self.data, batch_size=batchsize, shuffle=shuffle, num_workers=worker)

        self.name = 'Impress_Perceiver'
        self.logger = logger
        self.sample_interval = sample_interval
        self.n_checkpoints = n_checkpoints
        self.epoch = None
        self.grad_vis = grad_vis
        self.grad_vis_g_done = not grad_vis

        self.attn_toggle = False

    def train(self, models, losses, optimizers, imgs, gt):
        # Configure x
        real_imgs = Variable(imgs.type(Tensor).to(GPU.device))

        input_imgs = rearrange(real_imgs, 'b c h w -> b h w c')
        img_shape = input_imgs.shape

        perceiver, deceiver, discriminator = models
        # mse_loss, gmsd_loss, adversarial_loss = losses
        mse_loss, adversarial_loss = losses
        optimizer, optimizer_D = optimizers
        valid, fake = gt

        # -----------------
        #  Train Generator
        # -----------------

        optimizer.zero_grad()

        latents = perceiver(input_imgs)

        x_hat = deceiver(latents, out_shape=img_shape)

        adv_loss = adversarial_loss(discriminator(latents), valid)
        # embedding_loss = embedding_loss.mean()
        recon_loss = mse_loss(input_imgs, x_hat)

        x_hat = rearrange(x_hat, 'b h w c -> b c h w')
        # extra_recon_loss = gmsd_loss(x_hat, real_imgs)

        # self.logger.log_value_and_epoch_avg('embedding_loss_train', embedding_loss.item(), self.epoch, self.dataloader.batch_size)
        self.logger.log_value_and_epoch_avg('recon_loss_train', recon_loss.item(), self.epoch, self.dataloader.batch_size)
        # self.logger.log_value_and_epoch_avg('gmsd_loss_train', extra_recon_loss.item(), self.epoch, self.dataloader.batch_size)
        self.logger.log_value_and_epoch_avg('adv_loss_train', adv_loss.item(), self.epoch, self.dataloader.batch_size)

        # loss = embedding_loss + recon_loss + extra_recon_loss
        # loss = recon_loss + extra_recon_loss + adv_loss * 0.01
        loss = recon_loss + adv_loss * 0.1

        loss.backward()

        optimizer.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Sample noise as discriminator ground truth
        z = Tensor(np.random.normal(0, 1, (self.dataloader.batch_size, perceiver.latent_dim))).to(GPU.device)

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(latents.detach()), valid)
        fake_loss = adversarial_loss(discriminator(z), fake)
        d_loss = 0.5 * (real_loss + fake_loss) * 0.01

        d_loss.backward()
        optimizer_D.step()

        return loss, d_loss

    def __call__(self, setup: SetupStruct, epoch: int, menu):
        models, losses, optimizers, schedulers = setup
        [model.train() for model in models]
        scheduler, scheduler_D = schedulers

        self.epoch = epoch
        self.grad_vis_g_done = not self.grad_vis
        self.grad_vis_d_done = not self.grad_vis

        self.logger.log_value('lr_encoder/decoder', scheduler.get_last_lr()[0], Training.global_step)
        self.logger.log_value('lr_D', scheduler_D.get_last_lr()[0], Training.global_step)

        running_loss_D = 0.0
        running_loss = 0.0
        # running_acc = 0.0

        valid = Tensor(self.dataloader.batch_size, 1).to(GPU.device).fill_(1.0).requires_grad_(False)
        fake = Tensor(self.dataloader.batch_size, 1).to(GPU.device).fill_(0.0).requires_grad_(False)

        pbar = tqdm(self.dataloader)
        pbar.set_description('Epoch {}'.format(epoch))

        for i, imgs in enumerate(pbar):
            batches_done = epoch * len(self.data) + i
            loss, d_loss = self.train(models, losses, optimizers, imgs, [valid, fake])

            if self.sample_interval is not None and batches_done % self.sample_interval * self.dataloader.batch_size == 0:
                asyncio.run(self.sample_image(models, n_row=self.dataloader.batch_size, batches_done=batches_done, epoch=epoch))

            # log values
            self.logger.step(self.dataloader.batch_size)
            Training.global_step += self.dataloader.batch_size

            # calculate log values
            running_loss += loss.item() * self.dataloader.batch_size
            # running_acc += accuracy * self.dataloader.batch_size
            running_loss_D += d_loss.item() * self.dataloader.batch_size

            self.logger.log_value_and_epoch_avg('train_loss', loss.item(), self.epoch, self.dataloader.batch_size)
            self.logger.log_value_and_epoch_avg('train_d_loss', d_loss.item(), self.epoch, self.dataloader.batch_size)
            # self.logger.log_value('train_accuracy', accuracy, Training.global_step)

            # menu()

        # calculate log values
        epoch_loss_D = running_loss_D / len(self.data)
        epoch_loss = running_loss / len(self.data)
        # epoch_acc = running_acc / len(self.data)

        self.logger.log_value('train_epoch_loss', epoch_loss, Training.global_step)

        # self.logger.log_value('train_epoch_accuracy', epoch_acc, Training.global_step)
        logging.info('Training {} epoch {} Lr: {:.6f} Loss: {:.6f} D_Loss: {:.6f}'
                     .format('Impress_Deceiver', epoch, scheduler.get_last_lr()[0], epoch_loss, epoch_loss_D))

        if not self.grad_vis_g_done:
            helper.plot_grad_flow_lines(models[0].named_parameters(),
                                        "{}/data/Grad_flow_perceiver_{}.lines.png".format(self.logger.log_dir, self.epoch))
            helper.plot_grad_flow_lines(models[1].named_parameters(),
                                        "{}/data/Grad_flow_deceiver_{}.lines.png".format(self.logger.log_dir, self.epoch))
            helper.plot_grad_flow_lines(models[2].named_parameters(),
                                        "{}/data/Grad_flow_discriminator_{}.lines.png".format(self.logger.log_dir, self.epoch))
            self.grad_vis_g_done = True

        scheduler.step()
        scheduler_D.step()

        # do checkpointing
        if self.n_checkpoints is None or epoch % self.n_checkpoints == 0:
            helper.save_checkpoint(epoch, Training.global_step, models, optimizers, self.logger.log_dir,
                                   self.n_checkpoints)

        return  # do training

    @staticmethod
    def get_latent_grid(model, n_rows=8):
        vector_stack = torch.zeros((model.num_latents, 1, model.latent_dim, model.latent_dim))
        for vec in range(model.num_latents):
            vectors = model.latents[vec]
            vector_stack[vec, :, :, :] = vectors.squeeze(0)
        vector_grid = make_grid(vector_stack, nrow=model.num_latents // n_rows)
        return vector_grid

    async def sample_image(self, models, n_row, batches_done, epoch):
        """Saves a grid of generated data"""
        [model.eval() for model in models]
        with torch.no_grad():
            perceiver, deceiver, discriminator = models

            batch_size = self.dataloader.batch_size

            batch_start = ((batches_done * batch_size) % len(self.data))
            batch_end = ((batches_done * batch_size + batch_size) % len(self.data))

            imgs = self.data[batch_start:batch_end]
            imgs = torch.stack(imgs)
            imgs = imgs.to(GPU.device)

            batch = rearrange(imgs, 'b c h w -> b h w c')
            img_shape = batch.shape

            # self.logger.log_graph(vqvae, imgs.detach())

            enc_img, perceiver_attn_maps = perceiver.forward_with_attention_maps(batch)
            batch, deceiver_attn_maps = deceiver.forward_with_attention_maps(enc_img, img_shape)
            batch = rearrange(batch, 'b h w c -> b c h w')

            batch_grid = make_grid(batch.detach(), nrow=n_row // 2)
            input_grid = make_grid(imgs, nrow=n_row // 2)

            batch_stack = torch.stack([input_grid, batch_grid])
            img_grid = make_grid(batch_stack, nrow=1)
            img_grid = helper.normalize(img_grid)
            save_image(img_grid.data, "{}/data/Reconstruction_{}_{}.png".format(self.logger.log_dir, epoch, batches_done), nrow=n_row)
            # helper.save_image(img_grid, "{}/data/Reconstruction_{}.pil".format(self.logger.log_dir, batches_done))
            fig, subplots = helper.create_image_figure(img_grid.cpu(), 'Reconstructed')
            self.logger.add_figure('{}_Reconstruction_{}_{:0>10}'.format(self.name, epoch, batches_done), fig)

            # if not self.attn_toggle:
            #     for index, attn_map in enumerate(perceiver_attn_maps):
            #         # attn_grid = make_grid(attn_map.view(batch_size, perceiver.num_latents, *img_shape[1:3]))
            #         attn_map = 1 - attn_map
            #         # rearrange(attn_map, ' b l (h w) -> b l h w', h=img_shape[1])
            #         attn_grid = make_grid(make_grid(attn_map.view(batch_size, perceiver.num_latents, *img_shape[1:3]), pad_value=1).unsqueeze(1), pad_value=0.5)
            #         attn_grid = helper.normalize(attn_grid)
            #         save_image(attn_grid.data, "{}/data/perceiver_attention_layer[{}]_{}_{}.png".format(self.logger.log_dir, index, epoch, batches_done), nrow=n_row)
            #
            #     for index, attn_map in enumerate(deceiver_attn_maps):
            #         attn_map = 1 - attn_map
            #         attn_grid = make_grid(make_grid(attn_map.view(batch_size, perceiver.num_latents, *img_shape[1:3]), pad_value=1).unsqueeze(1), pad_value=0.5)
            #         attn_grid = helper.normalize(attn_grid)
            #         save_image(attn_grid.data, "{}/data/deceiver_attention_layer[{}]_{}_{}.png".format(self.logger.log_dir, index, epoch, batches_done), nrow=n_row)
            # self.attn_toggle = not self.attn_toggle

            # vector_grid = self.get_latent_grid(perceiver)
            # vector_grid = helper.normalize(vector_grid)
            # save_image(vector_grid.data, "{}/data/perceiver_latents_vis_{}_{}.png".format(self.logger.log_dir, epoch, batches_done), nrow=n_row)
            # fig, subplots = helper.create_image_figure(vector_grid.cpu(), 'Vector_visualization')
            # self.logger.add_figure('{}perceiver_latents_vis_{}_{:0>10}'.format(self.name, epoch, batches_done), fig)
            #
            # vector_grid = self.get_latent_grid(deceiver)
            # vector_grid = helper.normalize(vector_grid)
            # save_image(vector_grid.data, "{}/data/deceiver_latents_vis_{}_{}.png".format(self.logger.log_dir, epoch, batches_done), nrow=n_row)
            # fig, subplots = helper.create_image_figure(vector_grid.cpu(), 'Vector_visualization')
            # self.logger.add_figure('{}deceiver_latents_vis_{}_{:0>10}'.format(self.name, epoch, batches_done), fig)

            z = Tensor(np.random.normal(0, 1, (batch_size, deceiver.latent_dim))).to(GPU.device)
            gen_imgs = deceiver(z, out_shape=img_shape)
            gen_imgs = rearrange(gen_imgs, 'b h w c -> b c h w')

            save_image(gen_imgs.data, "{}/data/{}random_sample_{}.png".format(self.logger.log_dir, self.name, batches_done), nrow=n_row, normalize=True)
            img_grid = make_grid(gen_imgs.detach(), nrow=batch_size, normalize=True)
            fig, subplots = helper.create_image_figure(img_grid.cpu(), 'Generated', 1, 1)
            self.logger.add_figure('{}_Random_Sample_{}_{:0>10}'.format(self.name, self.epoch, batches_done), fig)
            plt.close()
        [model.train() for model in models]
