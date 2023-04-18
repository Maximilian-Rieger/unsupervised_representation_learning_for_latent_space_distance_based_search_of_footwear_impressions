import os
import asyncio
import datetime
from typing import Tuple, List

import faiss
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

from models.vqvae.vqvae import VQVAE_b as VQVAE, VQVAE_2, VQVAE_3, VQVAE_4

from einops import repeat, rearrange

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
                'lr': 0.001,
                'step_size': 500,
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
                'sample_interval': None,
                'batchsize': 1,
                'worker': 0,
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
        return f'{self.args["batchsize"]}x{self.args["in"]}x{self.args["out"]}'

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
        vqvae = VQVAE(
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
        # initialize model weights with xavier
        vqvae.apply(helper.weights_init)

        summary(vqvae.encoder, input_size=(vqvae.encoder.in_dim, 128, 128), batch_size=batch_size)
        summary(vqvae.decoder, input_size=(vqvae.decoder.in_dim, 15, 15), batch_size=batch_size)

        # recon_loss = torch.nn.SmoothL1Loss().to(GPU.device)
        recon_loss = torch.nn.MSELoss().to(GPU.device)

        return [[vqvae], [recon_loss]]

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
            batch_norm,
            batch_size
    ):
        models = Experiment.model(
            batch_size=batch_size,
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

        # optimizer_vqvae = AdaBelief(vqvae.parameters(), lr=lr, betas=(beta1, beta2))
        optimizer_vqvae = torch.optim.AdamW(vqvae.parameters(), lr=lr, betas=(beta1, beta2))

        scheduler_vqvae = lr_scheduler.StepLR(optimizer_vqvae, step_size=step_size, gamma=gamma)

        return [[optimizer_vqvae], [scheduler_vqvae]]

    def setup(self) -> SetupStruct:
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            # transforms.RandomResizedCrop(img_shape[1:], (0.8, 1.0)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

        # initialize training data
        self.training = Training(**self.args['training'], transform=self.transform, logger=self.logger)
        self.validation = Validation(**self.args['validation'], transform=self.transform, logger=self.logger)

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
            self.validation(setup=setup, epoch=epoch, menu=self.keyboard_menu)

        helper.save_checkpoint(end + 1, Training.global_step, setup[0], setup[2], self.logger.log_dir, self.args['training']['n_checkpoints'])

    @staticmethod
    def get_vector_grid(vqvae, batch_size, n_rows=16):
        vector_stack = torch.zeros((vqvae.vector_quantization.n_e, 1, 128, 128))
        # g = vqvae.encoder.conv_stack[0].kernel_size[0]
        g = 8
        for vec in range(0, vqvae.vector_quantization.n_e, batch_size):
            vectors = repeat(vqvae.vector_quantization.embedding.weight[vec:vec + batch_size], 'b d -> b d h w', h=g, w=g)
            vectors = Experiment.mask_border(vectors, g // 2 - 2)
            vectors = vqvae.decoder(vectors)
            vector_stack[vec:vec + batch_size, :, :, :] = vectors
        vector_grid = make_grid(vector_stack, nrow=vqvae.vector_quantization.n_e // n_rows)
        return vector_grid

    @staticmethod
    def mask_border(img, border=1):
        # mask bottom row
        img[:, :, -border:, :] = 0
        # mask top row
        img[:, :, :border, :] = 0
        # mask left column
        img[:, :, :, :border] = 0
        # mask right column
        img[:, :, :, -border:] = 0
        return img


class Training:
    global_step = 0

    def __init__(self, data, batchsize, shuffle, worker, sample_interval, n_checkpoints, logger, grad_vis, **kwargs):
        self.data = DataZoo.get(**data, **kwargs)
        self.dataloader = DataLoader(self.data, batch_size=batchsize, shuffle=shuffle, num_workers=worker)
        self.data_len = len(self.data)
        self.batch_size = batchsize
        self.batch_count = self.data_len // self.batch_size

        self.name = 'Impress_VQ-VAE'
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
        mse_loss, = losses
        optimizer_vqvae, = optimizers

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_vqvae.zero_grad()

        embedding_loss, x_hat = vqvae(real_imgs)[0:2]
        embedding_loss = embedding_loss.mean()
        recon_loss = mse_loss(real_imgs, x_hat)

        self.logger.log_value_and_epoch_avg('embedding_loss_train', embedding_loss.item(), self.epoch, self.batch_size, self.data_len)
        self.logger.log_value_and_epoch_avg('recon_loss_train', recon_loss.item(), self.epoch, self.batch_size, self.data_len)

        loss = embedding_loss + recon_loss

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

        # running_acc = 0.0

        pbar = tqdm(self.dataloader)
        pbar.set_description('Epoch {}'.format(epoch))

        scaled_sample_interval = self.sample_interval * self.batch_count

        for i, imgs in enumerate(pbar):
            batches_done = epoch * len(self.data) + i
            loss = self.train(models, losses, optimizers, imgs)
            self.logger.log_value_and_epoch_avg('train_loss', loss.item(), self.epoch, self.batch_size, self.data_len)

            if self.sample_interval is not None and batches_done % scaled_sample_interval == 0:
                self.sample_image(models, n_row=self.dataloader.batch_size, batches_done=batches_done)

            # log values
            self.logger.step(self.dataloader.batch_size)
            Training.global_step += self.dataloader.batch_size

            # calculate log values
            # running_acc += accuracy * self.dataloader.batch_size

            # self.logger.log_value('train_accuracy', accuracy, Training.global_step)

            # menu()

        # calculate log values
        epoch_loss = self.logger.get_epoch_loss('train_loss', self.data_len)
        self.logger.log_value('_epoch_training_loss_', epoch_loss,  step=self.epoch)
        # self.logger.log_value('train_epoch_accuracy', epoch_acc, Training.global_step)
        logging.info(f'Training Impress_VQ-VAE epoch {epoch} Lr: {scheduler_vqvae.get_last_lr()[0]:.8f} Loss: {epoch_loss:.4f}')

        if not self.grad_vis_g_done:
            helper.plot_grad_flow_lines(models[0].named_parameters(), f"{self.logger.log_dir}/data/Grad_flow_vqvae_{self.epoch}.lines.png")
            self.grad_vis_g_done = True

        scheduler_vqvae.step()

        # do checkpointing
        if self.n_checkpoints is None or epoch % self.n_checkpoints == 0:
            helper.save_checkpoint(epoch, Training.global_step, models, optimizers, self.logger.log_dir, self.n_checkpoints)

        return  # do training

    def sample_image(self, models, n_row, batches_done):
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

            # batch_grid = make_grid(batch.detach(), nrow=n_row // 16)
            # input_grid = make_grid(imgs, nrow=n_row // 16)
            b = torch.stack([*imgs, *batch])
            b_half = b.shape[0] // 2
            img_grid = make_grid(torch.cat([torch.cat([b[:b_half:2], b[b_half::2]], dim=2), torch.cat([b[1:b_half:2], b[b_half+1::2]], dim=2)]).detach().cpu(), nrow=n_row // 16)
            # save_image(img_grid.data, f"{self.logger.log_dir}/data/Reconstruction_{self.epoch}_{batches_done}_test.png", nrow=n_row)
            # batch_stack = torch.stack([input_grid, batch_grid])
            # img_grid = make_grid(batch_stack, nrow=1)
            # img_grid = helper.normalize(img_grid)
            save_image(img_grid.data, "{}/data/Reconstruction_{}_{}.png".format(self.logger.log_dir, self.epoch, batches_done), nrow=n_row)
            # helper.save_image(img_grid, "{}/data/Reconstruction_{}.pil".format(self.logger.log_dir, batches_done))
            # fig, subplots = helper.create_image_figure(img_grid.cpu(), 'Reconstructed')

            # self.logger.add_figure('{}_Reconstruction_{}_{:0>10}'.format(self.name, epoch, batches_done), fig)
            # fig.savefig("{}/data/Reconstruction_fig_{:0>10}.png".format(self.logger.log_dir, batches_done))

            vector_grid = Experiment.get_vector_grid(vqvae, self.batch_size)
            # vector_grid = helper.normalize(vector_grid)
            save_image(vector_grid.data, "{}/data/vector_vis_new_{}_{}.png".format(self.logger.log_dir, self.epoch, batches_done), nrow=n_row)
            # fig, subplots = helper.create_image_figure(vector_grid.cpu(), 'Vector_visualization')
            # self.logger.add_figure('{}_Vector_visualization_{}_new_{:0>10}'.format(self.name, epoch, batches_done), fig)
        [model.train() for model in models]


class Validation:
    global_step = 0

    def __init__(self, data, batchsize, worker, logger, sample_interval, save_best_model=True, **kwargs):
        self.data = DataZoo.get(**data, **kwargs)
        self.batch_size = batchsize
        self.dataloader = DataLoader(self.data, batch_size=batchsize, shuffle=False, num_workers=worker)
        self.name = 'Impress_VQ-VAE'
        self.logger = logger
        self.sample_interval = sample_interval
        self.epoch = None
        self.save_best_model = save_best_model
        self.best_loss = float('inf')
        self.best_kmeans_acc = float('inf')
        self.data_len = len(self.data)
        self.batch_count = self.data_len // self.batch_size
        self.batch_count = self.batch_count if self.batch_count > 0 else 1
        self.last_ca_epoch = 0

    def train(self, models, losses, data):
        # Configure x
        labels, imgs = data
        real_imgs = Variable(imgs.type(Tensor).to(GPU.device))

        vqvae, = models
        mse_loss, = losses

        # -----------------
        #  Validate Generator
        # -----------------

        embedding_loss, x_hat, _, latents = vqvae(real_imgs)
        self.logger.accumulate_embedding_set_for_epoch(latents.view(latents.shape[0], -1).cpu(), labels, images=imgs, name=self.name)
        embedding_loss = embedding_loss.mean()
        recon_loss = mse_loss(real_imgs, x_hat)

        self.logger.log_value_and_epoch_avg('embedding_loss_val', embedding_loss.item(), self.epoch, self.batch_size, self.data_len)
        self.logger.log_value_and_epoch_avg('recon_loss_val', recon_loss.item(), self.epoch, self.batch_size, self.data_len)

        loss = embedding_loss + recon_loss

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
            self.logger.log_value_and_epoch_avg('validation_loss', loss.item(), self.epoch, self.batch_size, self.data_len)

            if self.sample_interval is not None and batches_done % scaled_sample_interval == 0:
                self.sample_image(models, n_row=self.batch_size, batches_done=batches_done)
            # log values
            self.logger.step(self.batch_size)
            Validation.global_step += self.batch_size

            # calculate log values
            running_loss += loss.item() * self.batch_size

        epoch_loss = self.logger.get_epoch_loss('validation_loss', self.data_len)
        self.logger.log_value('_epoch_validation_loss_', epoch_loss, step=self.epoch)
        kmeans_acc = None
        if epoch - self.last_ca_epoch > 10 or epoch_loss < self.best_loss:
            features, labels, labels_header, images = self.logger.get_embedding_set(self.name)
            clusters = len(labels.unique())
            fmeans = faiss.Kmeans(features.shape[1], clusters)
            fmeans.train(features.detach().numpy())
            cluster_centers, cluster_ids_x = fmeans.assign(features.detach().numpy())
            kmeans_acc = 0.0
            for n in range(1, clusters + 1, 2):
                kmeans_acc += 2 if cluster_ids_x[n - 1] == cluster_ids_x[n] else 0
            kmeans_acc /= clusters + 1
            self.last_ca_epoch = epoch
            self.logger.log_value('Kmeans_acc_val', kmeans_acc)
            if self.best_kmeans_acc > kmeans_acc:
                self.best_kmeans_acc = kmeans_acc
                helper.save_checkpoint(epoch, Training.global_step, models, optimizers, self.logger.log_dir, custom_name='best_kmeans_acc')

        if epoch_loss < self.best_loss:
            self.best_loss = epoch_loss
            self.last_ca_epoch = epoch
            helper.save_checkpoint(epoch, Training.global_step, models, optimizers, self.logger.log_dir, best=True)

            self.logger.log_embedding_set(self.name, step=epoch)
        else:
            self.logger.clear_embedding_set(self.name)
        kmeans_acc = f"{kmeans_acc:.6f}" if kmeans_acc is not None else "N/A"
        logging.info(f'Validation {self.name} epoch {epoch} Loss: {epoch_loss:.6f} Clustering_acc: {kmeans_acc}')

        # do checkpointing
        return  # do training

    def sample_image(self, models, n_row, batches_done):
        """Saves a grid of generated data"""
        [model.eval() for model in models]
        with torch.no_grad():
            vqvae, = models

            batch_size = self.dataloader.batch_size

            batch_start = ((batches_done * batch_size) % len(self.data))
            batch_end = ((batches_done * batch_size + batch_size) % len(self.data))

            data = self.data[batch_start:batch_end]
            imgs = [img for _, img in data]
            imgs = torch.stack(imgs)
            imgs = imgs.to(GPU.device)

            # self.logger.log_graph(vqvae, imgs.detach())

            batch, _, _ = vqvae.reconstruct(imgs.detach())
            #
            # batch_grid = make_grid(batch.detach(), nrow=n_row // 16)
            # input_grid = make_grid(imgs, nrow=n_row // 16)
            #
            # batch_stack = torch.stack([input_grid, batch_grid])
            # img_grid = make_grid(batch_stack, nrow=1)
            b = torch.stack([*imgs, *batch])
            b_half = b.shape[0] // 2
            img_grid = make_grid(torch.cat([torch.cat([b[:b_half:2], b[b_half::2]], dim=2), torch.cat([b[1:b_half:2], b[b_half + 1::2]], dim=2)]).detach().cpu(), nrow=n_row // 16)
            # img_grid = helper.normalize(img_grid)
            save_image(img_grid.data, "{}/data/Reconstruction_val_{}_{}.png".format(self.logger.log_dir, self.epoch, batches_done), nrow=n_row)
            # helper.save_image(img_grid, "{}/data/Reconstruction_{}.pil".format(self.logger.log_dir, batches_done))

            # vector_grid = Experiment.get_vector_grid(vqvae, self.batch_size)
            # # vector_grid = helper.normalize(vector_grid)
            # save_image(vector_grid.data, "{}/data/vector_vis_new_{}_{}.png".format(self.logger.log_dir, self.epoch, batches_done), nrow=n_row)
            # # fig, subplots = helper.create_image_figure(vector_grid.cpu(), 'Vector_visualization')
            # # self.logger.add_figure('{}_Vector_visualization_{}_new_{:0>10}'.format(self.name, epoch, batches_done), fig)
        [model.train() for model in models]
