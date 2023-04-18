import os
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
import itertools

import experiment.helper as helper
from torchsummary import summary

from models.SupRVAE import ConvEncoder, ConvDecoder, Discriminator
from models.utils import Mish, SwishLayer

from optimizers.Ranger import Ranger
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
                'latent_size': 256,
                'encoder_depth': 2,
                'decoder_depth': 2,
                'discriminator_depth': 2,
                'residual_enc': False,
                'residual_dec': False,
                'start_pixels_dec': 64
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
        self.args = helper.forward_arguments(self.args, ['batchsize'])

        self.logger = TensorboardLogger(os.path.join(args['log_dir'], args['name']), modules=[__name__], images_dir=True)

        now = datetime.datetime.now()
        self.args['started_training'] = '%d-%02d-%02d-%02d-%02d' % (now.year, now.month, now.day, now.hour, now.minute)
        self.logger.log_options(self.args, changes)

        self.loader_args = {'num_workers': 8, 'pin_memory': False} if self.args['cuda'] else {}

    def __str__(self) -> str:
        return '{}x{}x{}'\
            .format(self.args['batchsize'], self.args['in'], self.args['out'])

    @staticmethod
    def model(img_shape, latent_size, encoder_depth, decoder_depth, discriminator_depth, residual_enc, residual_dec, start_pixels_dec) -> [[nn.Module], [nn.Module]]:
        activation = nn.ReLU
        encoder = ConvEncoder(img_shape, latent_size=latent_size, depth=encoder_depth, activation=activation, residual=residual_enc).to(GPU.device)
        decoder = ConvDecoder(
            img_shape, latent_size=latent_size, depth=decoder_depth, activation=activation,
            residual=residual_dec, start_pixels=start_pixels_dec, last=nn.Sigmoid
        ).to(GPU.device)
        discriminator = Discriminator(latent_size, depth=discriminator_depth, activation=activation).to(GPU.device)

        summary(encoder, input_size=img_shape, batch_size=4)
        summary(decoder, input_size=(latent_size,), batch_size=4)
        summary(discriminator, input_size=(latent_size,), batch_size=4)

        adversarial_loss = torch.nn.BCELoss().to(GPU.device)
        pixelwise_loss = torch.nn.BCELoss().to(GPU.device)
        # pixelwise_loss = torch.nn.SmoothL1Loss().to(GPU.device)
        discriminator_loss = torch.nn.BCELoss().to(GPU.device)

        return [[encoder, decoder, discriminator], [adversarial_loss, pixelwise_loss, discriminator_loss]]

    @staticmethod
    def optimizer(models, lr, step_size, beta1=0.9, beta2=0.99, gamma=0.1):
        encoder, decoder, discriminator = models
        optimizer_Enc = torch.optim.Adam(encoder.parameters(), lr=lr, betas=(beta1, beta2))
        optimizer_Dec = torch.optim.Adam(decoder.parameters(), lr=lr, betas=(beta1, beta2))
        optimizer_Dis = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

        scheduler_Enc = lr_scheduler.StepLR(optimizer_Enc, step_size=step_size, gamma=gamma)
        scheduler_Dec = lr_scheduler.StepLR(optimizer_Dec, step_size=step_size, gamma=gamma)
        scheduler_Dis = lr_scheduler.StepLR(optimizer_Dis, step_size=step_size, gamma=gamma)

        return [[optimizer_Enc, optimizer_Dec, optimizer_Dis], [scheduler_Enc, scheduler_Dec, scheduler_Dis]]

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
        self.validate = Validate(**self.args['training'], transform=self.transform, logger=self.logger)

        models, losses = Experiment.model(img_shape, **self.args['model'])
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
        self.grad_vis_d_done = not grad_vis

    def train_g(self, models: List[nn.Module], losses: List[nn.Module], optimizers, input, gt_vec, batches_done, adv_weight=0.5):
        encoder, decoder, discriminator = models
        adversarial_loss, pixelwise_loss, _ = losses
        optimizer_Enc, optimizer_Dec = optimizers[0:2]
        valid, fake = gt_vec

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_Enc.zero_grad()
        optimizer_Dec.zero_grad()

        encoded_imgs = encoder(input)
        decoded_imgs = decoder(encoded_imgs)

        # Loss measures generator's ability to fool the discriminator
        # print()
        g_forging_loss = adversarial_loss(discriminator(encoded_imgs), valid)
        # logging.info('Forging loss {:.10f}'.format(g_forging_loss))
        g_reconstruction_loss = pixelwise_loss(decoded_imgs, input)
        # logging.info('Reconstruction loss {:.10f}'.format(g_reconstruction_loss))

        g_loss = adv_weight * g_forging_loss + (1 - adv_weight) * g_reconstruction_loss
        # logging.info('Combined loss {:.10f}'.format(g_loss))

        g_loss.backward()

        optimizer_Enc.step()
        optimizer_Dec.step()

        # encoder_grad = helper.mean_grad(encoder.named_parameters())
        # logging.info('Mean encoder grad: {:.10f}'.format(encoder_grad))
        # decoder_grad = helper.mean_grad(decoder.named_parameters())
        # logging.info('Mean decoder grad: {:.10f}'.format(decoder_grad))

        if not self.grad_vis_g_done:
            helper.plot_grad_flow_lines(encoder.named_parameters(), "{}/data/Grad_flow_enc_{}.lines.png".format(self.logger.log_dir, self.epoch))
            helper.plot_grad_flow_lines(decoder.named_parameters(), "{}/data/Grad_flow_dec_{}.lines.png".format(self.logger.log_dir, self.epoch))
            self.grad_vis_g_done = True

        return g_loss, encoded_imgs

    def accuracy(self, pred, target):
        return (pred > 0.5).type(torch.cuda.FloatTensor).eq(target).sum().item() / self.dataloader.batch_size

    def train_d(self, models: List[nn.Module], losses: List[nn.Module], optimizers, input, gt_vec: List[torch.autograd.Variable], batches_done):
        encoder, decoder, discriminator = models
        discriminator_loss = losses[2]
        optimizer_D = optimizers[2]
        valid, fake = gt_vec
        real_imgs, encoded_imgs = input

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as discriminator ground truth
        z = Variable(Tensor(np.random.normal(0, 1, (self.dataloader.batch_size, encoder.latent_size))).to(GPU.device))

        # Measure discriminator's ability to classify real from generated samples
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
        optimizer_D.step()
        # logging.info('Mean discriminator grad: {:.10f}'.format(helper.mean_grad(discriminator.named_parameters())))

        if not self.grad_vis_d_done:
            helper.plot_grad_flow_lines(discriminator.named_parameters(), "{}/data/Grad_flow_dis_{}.lines.png".format(self.logger.log_dir, self.epoch))
            self.grad_vis_d_done = True

        return d_loss, accuracy

    def train(self, models, losses, optimizers, imgs, gt_vec, batches_done):
        # Configure x
        real_imgs = Variable(imgs.type(Tensor).to(GPU.device))

        g_loss, encoded_imgs = self.train_g(models, losses, optimizers, real_imgs, gt_vec, batches_done)

        d_loss, accuracy = self.train_d(models, losses, optimizers, [real_imgs, encoded_imgs], gt_vec, batches_done)

        return g_loss, d_loss, accuracy

    def __call__(self, setup: SetupStruct, epoch: int, menu):
        models, losses, optimizers, schedulers = setup
        [model.train() for model in models]
        # [scheduler.step() for scheduler in schedulers]
        scheduler_enc, scheduler_dec, scheduler_d = schedulers
        scheduler_enc.step(); scheduler_dec.step(); scheduler_d.step()

        self.epoch = epoch
        self.grad_vis_g_done = not self.grad_vis
        self.grad_vis_d_done = not self.grad_vis

        self.logger.log_value('lr_enc', scheduler_enc.get_lr()[0], Training.global_step)
        self.logger.log_value('lr_dec', scheduler_dec.get_lr()[0], Training.global_step)
        self.logger.log_value('lr_dis', scheduler_d.get_lr()[0], Training.global_step)

        running_loss_D = 0.0
        running_loss_G = 0.0
        running_acc_D = 0.0

        pbar = tqdm(self.dataloader)
        pbar.set_description('Epoch {}'.format(epoch))

        # Adversarial ground truths
        valid = Variable(Tensor(self.dataloader.batch_size, 1).to(GPU.device).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(self.dataloader.batch_size, 1).to(GPU.device).fill_(0.0), requires_grad=False)

        for i, imgs in enumerate(pbar):
            batches_done = epoch * len(self.data) + i
            g_loss, d_loss, accuracy = self.train(models, losses, optimizers, imgs, [valid, fake], batches_done)

            if self.sample_interval is not None and batches_done % self.sample_interval == 0:
                self.sample_image(models, n_row=self.dataloader.batch_size, batches_done=batches_done)

            # log values
            self.logger.step(self.dataloader.batch_size)
            Training.global_step += self.dataloader.batch_size

            # calculate log values
            running_loss_G += g_loss.item() * self.dataloader.batch_size
            running_loss_D += d_loss.item() * self.dataloader.batch_size
            running_acc_D += accuracy * self.dataloader.batch_size

            self.logger.log_value('train_d_loss', d_loss.item(), Training.global_step)
            self.logger.log_value('train_g_loss', g_loss.item(), Training.global_step)
            self.logger.log_value('train_d_accuracy', accuracy, Training.global_step)

            #menu()

        # calculate log values
        epoch_loss_G = running_loss_G / len(self.data)
        epoch_loss_D = running_loss_D / len(self.data)
        epoch_acc_D = running_acc_D / len(self.data)

        self.logger.log_value('train_epoch_g_loss', epoch_loss_G, Training.global_step)
        self.logger.log_value('train_epoch_d_loss', epoch_loss_D, Training.global_step)
        self.logger.log_value('train_epoch_d_acc', epoch_acc_D, Training.global_step)
        logging.info('Training {} epoch {} Loss_G: {:.4f} Loss_D: {:.4f} Accuracy_D: {:.4}'
                     .format('Impress_SVRAE', epoch, epoch_loss_G, epoch_loss_D, epoch_acc_D))

        # do checkpointing
        if self.n_checkpoints is None or epoch % self.n_checkpoints == 0:
            helper.save_checkpoint(epoch, Training.global_step, models, optimizers, self.logger.log_dir, self.n_checkpoints)


        return  # do training

    def sample_image(self, models, n_row, batches_done):
        """Saves a grid of generated data"""
        # [model.eval() for model in models]
        encoder, decoder, discriminator = models
        # Sample noise
        z = Variable(Tensor(np.random.normal(0, 1, (n_row, decoder.latent_size))).to(GPU.device))
        gen_imgs = decoder(z)
        gen_imgs = helper.normalize(gen_imgs)
        batch_size = self.dataloader.batch_size

        save_image(gen_imgs.data, "{}/data/sample_{}.png".format(self.logger.log_dir, batches_done), nrow=n_row)
        # helper.save_image(gen_imgs, "{}/data/sample_{}.pil".format(self.logger.log_dir, batches_done))
        img_grid = make_grid(gen_imgs.detach(), nrow=n_row)
        fig, subplots = helper.create_image_figure(img_grid.cpu(), 'Generated', 1, 1)
        self.logger.add_figure('{}_Sample_{:0>10}'.format('Impress_SupVAE', batches_done), fig)
        fig.savefig("{}/data/Sample_fig_{:0>10}.png".format(self.logger.log_dir, batches_done))

        batch_start = ((batches_done * batch_size) % len(self.data))
        batch_end = ((batches_done * batch_size + batch_size) % len(self.data))

        imgs = self.data[batch_start:batch_end]
        imgs = torch.stack(imgs)
        imgs = imgs.to(GPU.device)

        # self.logger.log_graph(encoder, imgs.detach())

        batch = encoder(imgs.detach()).detach()
        # self.logger.log_graph(decoder, batch.detach())

        # plt.scatter(batch[:, 0], batch[:, 1], c=[i for i in range(self.dataloader.batch_size)])
        # plt.savefig("{}/data/Scatter_fig_{:0>10}.png".format(self.logger.log_dir, batches_done))
        prediction = discriminator(batch.detach()).detach()
        pixels = np.prod(gen_imgs.shape[1:])
        prediction = prediction.repeat(1, pixels).view(*gen_imgs.shape)
        batch = decoder(batch)
        batch_grid = make_grid(batch.detach(), nrow=n_row)
        input_grid = make_grid(imgs, nrow=n_row)
        prediction_grid = make_grid(prediction, nrow=n_row)


        batch_stack = torch.stack([input_grid, batch_grid, prediction_grid])
        img_grid = make_grid(batch_stack, nrow=1)
        img_grid = helper.normalize(img_grid)
        save_image(img_grid.data, "{}/data/Reconstruction_{}.png".format(self.logger.log_dir, batches_done), nrow=n_row)
        # helper.save_image(img_grid, "{}/data/Reconstruction_{}.pil".format(self.logger.log_dir, batches_done))
        fig, subplots = helper.create_image_figure(img_grid.cpu(), 'Reconstructed')
        self.logger.add_figure('{}_Reconstruction_{:0>10}'.format('Impress_SupVAE', batches_done), fig)
        # fig.savefig("{}/data/Reconstruction_fig_{:0>10}.png".format(self.logger.log_dir, batches_done))
        # [model.train() for model in models]


class Validate:
    global_step = 0

    def __init__(self, data, batchsize, shuffle, worker, sample_interval, n_checkpoints, logger, grad_vis, **kwargs):
        self.data = DataZoo.get(**data, **kwargs)
        self.dataloader = DataLoader(self.data, batch_size=batchsize, shuffle=shuffle, num_workers=worker)

        self.logger = logger
        self.sample_interval = sample_interval
        self.n_checkpoints = n_checkpoints
        self.grad_vis = grad_vis

    def accuracy(self, pred, target):
        return (pred > 0.5).type(torch.cuda.FloatTensor).eq(target).sum().item() / self.dataloader.batch_size

    def eval(self, models, losses, optimizers, imgs, gt_vec):
        # Configure x
        real_imgs = Variable(imgs.type(Tensor).to(GPU.device))

        g_loss, encoded_imgs = self.train_g(models, losses, optimizers, real_imgs, gt_vec)

        d_loss, accuracy = self.train_d(models, losses, optimizers, [real_imgs, encoded_imgs], gt_vec)

        return g_loss, d_loss, accuracy

    def __call__(self, setup: SetupStruct, epoch: int, menu):
        models, _, _, _ = setup
        [model.eval() for model in models]
        encoder, _, _ = models

        pbar = tqdm(self.dataloader)
        pbar.set_description('Epoch {}'.format(epoch))

        for i, imgs in enumerate(pbar):
            real_imgs = Variable(imgs.type(Tensor).to(GPU.device))
            encoded_batch = encoder(real_imgs).detach().cpu()
            plt.scatter(encoded_batch[:, 0], encoded_batch[:, 1], c=[n for n in range(self.dataloader.batch_size)])

            batches_done = epoch * len(self.data) + i
            if batches_done % self.sample_interval == 0:
                self.sample_image(models, n_row=self.dataloader.batch_size, batches_done=batches_done)

            # log values
            self.logger.step(self.dataloader.batch_size)
            Validate.global_step += self.dataloader.batch_size

            #menu()

        plt.savefig("{}/data/Full_Scatter_fig_{:0>3}.val.png".format(self.logger.log_dir, epoch))
        logging.info('Did full scatter plot for Epoch {}'.format(epoch))
        return  # do training

    def sample_image(self, models, n_row, batches_done):
        """Saves a grid of generated data"""
        encoder, decoder, discriminator = models
        # Sample noise
        z = Variable(Tensor(np.random.normal(0, 1, (n_row, decoder.latent_size))).to(GPU.device))
        gen_imgs = decoder(z)

        save_image(gen_imgs.data, "{}/data/sample_{}.val.png".format(self.logger.log_dir, batches_done), nrow=n_row, normalize=True)
        img_grid = make_grid(gen_imgs.detach(), nrow=n_row, normalize=True)
        fig, subplots = helper.create_image_figure(img_grid.cpu(), 'Generated', 1, 1)
        self.logger.add_figure('{}_Sample_{:0>10}'.format('Impress_SupVAE', batches_done), fig)
        fig.savefig("{}/data/Sample_fig_{:0>10}.val.png".format(self.logger.log_dir, batches_done))

        batch_size = self.dataloader.batch_size
        batch_start = ((batches_done * batch_size) % len(self.data))
        batch_end = ((batches_done * batch_size + batch_size) % len(self.data))

        imgs = self.data[batch_start:batch_end]
        imgs = torch.stack(imgs)
        imgs = imgs.to(GPU.device)

        batch = encoder(imgs.detach()).detach()
        prediction = discriminator(batch.detach()).detach()
        pixels = np.prod(gen_imgs.shape[1:])
        prediction = prediction.repeat(1, pixels).view(*gen_imgs.shape)
        batch = decoder(batch)
        batch_grid = make_grid(batch.detach(), nrow=n_row, normalize=True)
        input_grid = make_grid(imgs, nrow=n_row, normalize=True)
        prediction_grid = make_grid(prediction, nrow=n_row, normalize=True)

        batch_stack = torch.stack([input_grid, batch_grid, prediction_grid])
        img_grid = make_grid(batch_stack, nrow=1, normalize=True)
        save_image(img_grid.data, "{}/data/Reconstruction_{}.png".format(self.logger.log_dir, batches_done), nrow=n_row, normalize=True)
        fig, subplots = helper.create_image_figure(img_grid.cpu(), 'Reconstructed')
        self.logger.add_figure('{}_Reconstruction_{:0>10}'.format('Impress_SupVAE', batches_done), fig)
        fig.savefig("{}/data/Reconstruction_fig_{:0>10}.val.png".format(self.logger.log_dir, batches_done))
