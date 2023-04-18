import os
import datetime
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
from experiment.data_zoo import DataZoo
import traceback

from torchsummary import summary

from utils.utils import TensorboardLogger
from utils.utils import dict_merge
from utils.utils import GPU
from utils.keyboard_menu import KeyboardMenu
import itertools

import experiment.helper as helper

from models.AAE import \
    ConvEncoder3 as ConvEncoder, \
    ConvDecoder3 as ConvDecoder, \
    Discriminator2 as Discriminator

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

            'img_size': 0,
            'latent_size': 0,
            'channels': 0,

            'optimizer': {
                'lr': 0.00001,
                'step_size': 5,
                'beta1': 0.9,
                'beta2': 0.99,
                'gamma': 0.1
            },

            'training': {
                'data': None,

                'batchsize': 1,
                'shuffle': True,
                'worker': 0,
                'sample_interval': None,
                'n_checkpoints': None,
                'log_image_info': None,
            },

            'validation': {
                'data': None,

                'batchsize': 1,
                'worker': 0,
                'sample_interval': None,
                'log_image_info': None,
            }
        }

        self.keyboard_menu = None
        self.training = None
        self.validation = None

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
        return '{}x{}x{}'\
            .format(self.args['batchsize'], self.args['in'], self.args['out'])

    @staticmethod
    def model(img_shape, latent_size, losses=True, show_summary=True) -> [[nn.Module], [nn.Module]]:
        activation = nn.LeakyReLU
        last = nn.Sigmoid
        batch_size = 6
        encoder = ConvEncoder(img_shape, latent_size=latent_size, activation=activation, activation_args={"negative_slope": 0.1}).to(GPU.device)
        decoder = ConvDecoder(img_shape, latent_size=latent_size, activation=activation, activation_args={"negative_slope": 0.1}, last=last).to(GPU.device)
        discriminator = Discriminator(latent_size=latent_size, activation=activation, activation_args={"negative_slope": 0.1}).to(GPU.device)

        if show_summary:
            summary(encoder, input_size=img_shape, batch_size=batch_size)
            summary(decoder, input_size=(latent_size,), batch_size=batch_size)
            summary(discriminator, input_size=(latent_size,), batch_size=batch_size)

        if not losses:
            return [[encoder, decoder, discriminator], []]

        adversarial_loss = torch.nn.BCELoss().to(GPU.device)
        pixelwise_loss = torch.nn.SmoothL1Loss().to(GPU.device)

        return [[encoder, decoder, discriminator], [adversarial_loss, pixelwise_loss]]

    @staticmethod
    def optimizer(models, lr, step_size, beta1=0.9, beta2=0.99, gamma=0.1):
        lr_g, lr_d = lr
        optimizer_G = torch.optim.Adam(itertools.chain(models[0].parameters(), models[1].parameters()), lr=lr_g, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(models[2].parameters(), lr=lr_d, betas=(beta1, beta2))

        scheduler_G = lr_scheduler.StepLR(optimizer_G, step_size=step_size, gamma=gamma)
        scheduler_D = lr_scheduler.StepLR(optimizer_D, step_size=step_size, gamma=gamma)
        return [[optimizer_G, optimizer_D], [scheduler_G, scheduler_D]]

    @staticmethod
    def load_model(path, img_shape, latent_size, show_summary=True):
        img_shape = (1, img_shape, img_shape)
        models = Experiment.model(img_shape, latent_size, show_summary=show_summary)[0]
        models, _, args, _ = helper.load_checkpoint(models, None, args={'resume': path})
        return models

    def setup(self):
        img_shape = (self.args['channels'], self.args['img_size'], self.args['img_size'])
        self.transform = transforms.Compose([
            transforms.Resize(img_shape[1:]),
            transforms.Grayscale(),
            transforms.RandomChoice([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]),
            transforms.ToTensor(),
        ])

        # initialize training data
        self.training = Training(**self.args['training'], transform=self.transform, logger=self.logger)
        if self.args['validation']['data'] is not None:
            self.validation = Validation(**self.args['validation'], transform=self.transform, logger=self.logger)

        models, criterions = Experiment.model(img_shape, self.args['latent_size'])
        optimizers, schedulers = Experiment.optimizer(models, **self.args['optimizer'])

        if self.args['resume'] is not None:
            models, optimizers, train_step = helper.load_checkpoint(models, optimizers, args=self.args)
            Training.global_step = train_step

        return models, criterions, optimizers, schedulers

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
        try:
            for epoch in range(start, end + 1):
                self.logger.epoch = epoch
                self.training(setup=setup, epoch=epoch, menu=self.keyboard_menu)
                if self.validation is not None:
                    self.validation(setup=setup, epoch=epoch, menu=self.keyboard_menu)
        except Exception:
            logging.getLogger().error(traceback.format_exc())


# noinspection PyArgumentList
class Training:
    global_step = 0

    def __init__(self, data, batchsize, shuffle, worker, sample_interval, n_checkpoints, logger, **kwargs):
        self.data = DataZoo.get(**data, **kwargs)
        self.dataloader = DataLoader(self.data, batch_size=batchsize, shuffle=shuffle, num_workers=worker)
        self.name = '{}-{}'.format(data['dataset'], data['set'])

        self.logger = logger
        self.sample_interval = sample_interval
        self.n_checkpoints = n_checkpoints
        self.epoch = None

    def accuracy(self, pred, target):
        return (pred > 0.5).type(torch.cuda.FloatTensor).eq(target).sum().item() / self.dataloader.batch_size

    def __call__(self, setup, epoch, menu):
        self.epoch = epoch
        models, criterions, optimizers, schedulers = setup
        [model.train() for model in models]

        encoder, decoder, discriminator = models
        adversarial_loss, pixelwise_loss = criterions
        optimizer_G, optimizer_D = optimizers

        running_accuracy = 0.0
        running_loss_D = 0.0
        running_loss_G = 0.0
        adversarial_loss_weight = 0.5


        grad_vis_g_done = False
        grad_vis_d_done = False
        pbar = tqdm(self.dataloader)
        pbar.set_description('Training Epoch {}'.format(epoch))
        for i, imgs in enumerate(pbar):
            # Adversarial ground truths
            valid = Tensor(imgs.shape[0], 1).to(GPU.device).fill_(1.0)
            fake = Tensor(imgs.shape[0], 1).to(GPU.device).fill_(0.0)

            # Configure x
            real_imgs = imgs.type(Tensor).to(GPU.device)
            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            encoded_imgs = encoder(real_imgs)
            decoded_imgs = decoder(encoded_imgs)
            # Loss measures generator's ability to fool the discriminator
            adv_loss = adversarial_loss(discriminator(encoded_imgs), valid)
            rec_loss = pixelwise_loss(decoded_imgs, real_imgs)
            g_loss = adversarial_loss_weight * adv_loss + (1 - adversarial_loss_weight) * rec_loss

            g_loss.backward()
            optimizer_G.step()
            # scheduler_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as discriminator ground truth
            z = Tensor(np.random.normal(0, 1, (imgs.shape[0], encoder.latent_size))).to(GPU.device)

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(encoded_imgs.detach()), valid)
            fake_loss = adversarial_loss(discriminator(z), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            optimizer_D.step()

            prediction = discriminator(encoded_imgs.detach())
            accuracy_valid = self.accuracy(prediction.detach(), valid)

            prediction = discriminator(z)
            accuracy_fake = self.accuracy(prediction.detach(), fake)

            accuracy = 0.5 * (accuracy_fake + accuracy_valid)

            # scheduler_D.step()

            if not grad_vis_g_done:
                param_list = {
                    'vae': itertools.chain(encoder.named_parameters(), decoder.named_parameters()),
                    'enc_dis': itertools.chain(encoder.named_parameters(), discriminator.named_parameters())
                }
                helper.plot_multiple_grad_flow_lines(param_list, "{}/data/Grad_flow_{}.png".format(self.logger.log_dir, epoch))
                grad_vis_g_done = True

            batches_done = epoch * len(self.data) + i
            if batches_done % self.sample_interval == 0:
                self.sample_image(models, batches_done)

            # log values
            self.logger.step(self.dataloader.batch_size)
            Training.global_step += self.dataloader.batch_size

            # calculate log values
            running_loss_G += g_loss.item() * imgs.size(0)
            running_loss_D += d_loss.item() * imgs.size(0)
            running_accuracy += accuracy * self.dataloader.batch_size

            self.logger.log_value('train_d_loss', d_loss.item(), Training.global_step)
            self.logger.log_value('train_g_loss', g_loss.item(), Training.global_step)
            # self.logger.log_value('train_accuracy', running_accuracy, Training.global_step)
            #menu()

        # calculate log values
        epoch_loss_G = running_loss_G / len(self.data)
        epoch_loss_D = running_loss_D / len(self.data)
        epoch_accuracy = running_accuracy / len(self.data)

        self.logger.log_value('train_epoch_g_loss', epoch_loss_G, Training.global_step)
        self.logger.log_value('train_epoch_d_loss', epoch_loss_D, Training.global_step)
        logging.info('Training {} epoch {} Loss_g: {:.6f} Loss_d: {:.6f} Accuracy: {:.4f}'
                     .format('Impress_AAE', epoch, epoch_loss_G, epoch_loss_D, epoch_accuracy))
        # logging.info('Training {} epoch {} Loss_g: {:.6f} Loss_d: {:.6f}'
        #              .format('Impress_AAE', epoch, epoch_loss_G, epoch_loss_D))
        [scheduler.step() for scheduler in schedulers]
        # do checkpointing
        helper.save_checkpoint(epoch, Training.global_step, models, optimizers, self.logger.log_dir, self.n_checkpoints)
        return  # do training

    def sample_image(self, models, batches_done, skewed_samples=False):
        """Saves a grid of generated data"""
        encoder, decoder, discriminator = models
        # Sample noise
        batch_size = self.dataloader.batch_size
        z = Tensor(np.random.normal(0, 1, (batch_size, decoder.latent_size))).to(GPU.device)
        nrow_min = min(batch_size, 4)
        gen_imgs = decoder(z)
        save_image(gen_imgs.data, "{}/data/random_sample_{}.png".format(self.logger.log_dir, batches_done), nrow=nrow_min, normalize=True)
        img_grid = make_grid(gen_imgs.detach(), nrow=batch_size, normalize=True)
        if img_grid.max() > 1 or img_grid.min() < 0:
            logging.getLogger().info('Img_grid[{}] value out of range 0-1'.format(batches_done))
        fig, subplots = helper.create_image_figure(img_grid.cpu(), 'Generated', 1, 1)
        self.logger.add_figure('Impress_AAE_{}_Random_Sample_{}_{:0>10}'.format(self.name, self.epoch, batches_done), fig)
        plt.close()
        batch_start = ((batches_done * batch_size) % len(self.data))
        batch_end = ((batches_done * batch_size + batch_size) % len(self.data))

        imgs = self.data[batch_start:batch_end]
        imgs = torch.stack(imgs)
        imgs = imgs.to(GPU.device)

        batch = encoder(imgs.detach())
        if skewed_samples:
            skewed_encoding = batch.detach()
            for i in range(1, batch_size):
                skewed_encoding[i] += skewed_encoding[0]
            skewed_images = decoder(skewed_encoding)
            save_image(skewed_images.data, "{}/data/skewed_sample_{}_{}.png".format(self.logger.log_dir, self.epoch, batches_done), nrow=batch_size, normalize=True)

        prediction = discriminator(batch.detach()).detach()
        prediction_bool = (prediction > 0.5).type(torch.cuda.FloatTensor)
        pixels = np.prod(decoder.img_shape)
        prediction = prediction.repeat(1, int(pixels / 2)).view(
            (batch_size, decoder.img_shape[0], int(decoder.img_shape[1] / 2), decoder.img_shape[2]))
        prediction_bool = prediction_bool.repeat(1, int(pixels / 2)).view(
            (batch_size, decoder.img_shape[0], int(decoder.img_shape[1] / 2), decoder.img_shape[2]))

        batch = decoder(batch)
        batch_grid = make_grid(batch.detach(), nrow=batch_size)
        input_grid = make_grid(imgs, nrow=batch_size)
        prediction_comb = torch.cat([prediction_bool, prediction], 2)
        prediction_grid = make_grid(prediction_comb, nrow=batch_size)

        # batch_stack = torch.stack([input_grid, batch_grid, input_grid - batch_grid, prediction_grid])
        batch_stack = torch.stack([input_grid, batch_grid, prediction_grid])
        img_grid = make_grid(batch_stack, nrow=1)
        img_grid = helper.normalize_img_tensor(img_grid)
        if img_grid.max() > 1 or img_grid.min() < 0:
            logging.getLogger().info('Img_grid[{}] value out of range 0-1'.format(batches_done))
        save_image(img_grid.data, "{}/data/Reconstruction_{}_{}.png".format(self.logger.log_dir, self.epoch, batches_done), nrow=batch_size, normalize=True)
        fig, subplots = helper.create_image_figure(img_grid.cpu(), 'Reconstructed')
        self.logger.add_figure('Impress_AAE_{}_Reconstruction_{}_{:0>10}'.format(self.name, self.epoch, batches_done), fig)
        plt.close()


class Validation:
    global_step = 0

    def __init__(self, data, batchsize, worker, logger, sample_interval, **kwargs):
        self.data = DataZoo.get(**data, **kwargs)

        self.dataloader = DataLoader(self.data, batch_size=batchsize, shuffle=False, num_workers=worker)
        self.name = '{}-{}'.format(data['dataset'], data['set'])
        self.logger = logger
        self.sample_interval = sample_interval
        self.epoch = None

    def accuracy(self, pred, target):
        return (pred > 0.5).type(torch.cuda.FloatTensor).eq(target).sum().item() / self.dataloader.batch_size

    def __call__(self, setup, epoch, menu):
        self.epoch = epoch
        models, criterions, _, _ = setup
        [model.eval() for model in models]

        encoder, decoder, discriminator = models
        adversarial_loss, pixelwise_loss = criterions

        running_accuracy = 0.0
        running_loss_D = 0.0
        running_loss_G = 0.0
        adversarial_loss_weight = 0.5

        pbar = tqdm(self.dataloader)
        pbar.set_description('Validation Epoch {}'.format(epoch))
        for i, imgs in enumerate(pbar):
            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1).to(GPU.device).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).to(GPU.device).fill_(0.0), requires_grad=False)

            # Configure x
            real_imgs = Variable(imgs.type(Tensor).to(GPU.device))

            # -----------------
            #  Train Generator
            # -----------------

            encoded_imgs = encoder(real_imgs)
            decoded_imgs = decoder(encoded_imgs)

            # Loss measures generator's ability to fool the discriminator
            adv_loss = adversarial_loss(discriminator(encoded_imgs), valid)
            rec_loss = pixelwise_loss(decoded_imgs, real_imgs)
            g_loss = adversarial_loss_weight * adv_loss + (1 - adversarial_loss_weight) * rec_loss

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Sample noise as discriminator ground truth
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], encoder.latent_size))).to(GPU.device))

            # Measure discriminator's ability to classify real from generated samples
            prediction = discriminator(encoded_imgs.detach())
            real_loss = adversarial_loss(prediction, valid)
            accuracy_valid = self.accuracy(prediction.detach(), valid)

            prediction = discriminator(z)
            fake_loss = adversarial_loss(prediction, fake)
            accuracy_fake = self.accuracy(prediction.detach(), fake)
            d_loss = 0.5 * (real_loss + fake_loss)
            accuracy = 0.5 * (accuracy_fake + accuracy_valid)

            batches_done = epoch * len(self.data) + i
            if batches_done % self.sample_interval == 0:
                self.sample_image(models, n_row=self.dataloader.batch_size, batches_done=batches_done)

            # log values
            self.logger.step(self.dataloader.batch_size)
            Training.global_step += self.dataloader.batch_size

            # calculate log values
            running_loss_G += g_loss.item() * imgs.size(0)
            running_loss_D += d_loss.item() * imgs.size(0)
            running_accuracy += accuracy * self.dataloader.batch_size

            self.logger.log_value('validation_d_loss', d_loss.item(), Training.global_step)
            self.logger.log_value('validation_g_loss', g_loss.item(), Training.global_step)

            # menu()

        # calculate log values
        epoch_accuracy = running_accuracy / len(self.data)
        epoch_loss_G = running_loss_G / len(self.data)
        epoch_loss_D = running_loss_D / len(self.data)

        self.logger.log_value('validation_epoch_g_loss', epoch_loss_G, Training.global_step)
        self.logger.log_value('validation_epoch_d_loss', epoch_loss_D, Training.global_step)
        logging.info('Validation {} epoch {} Loss_g: {:.6f} Loss_d: {:.6f} Accuracy: {:.6f}'
                     .format('Impress_AAE', epoch, epoch_loss_G, epoch_loss_D, epoch_accuracy))

        # do checkpointing
        return  # do training

    def sample_image(self, models, n_row, batches_done):
        """Saves a grid of generated data"""
        encoder, decoder, discriminator = models
        # Sample noise
        batch_size = self.dataloader.batch_size
        batch_start = ((batches_done * batch_size) % len(self.data))
        batch_end = ((batches_done * batch_size + batch_size) % len(self.data))

        imgs = self.data[batch_start:batch_end]
        imgs = torch.stack(imgs)
        imgs = imgs.to(GPU.device)

        batch = encoder(imgs.detach())
        prediction = discriminator(batch.detach()).detach()
        prediction = prediction.repeat(1, 3 * 256 ** 2).view(self.dataloader.batch_size, 3, 256, 256)
        batch = decoder(batch)
        batch_grid = make_grid(batch.detach(), nrow=n_row)
        input_grid = make_grid(imgs, nrow=n_row)
        prediction_grid = make_grid(prediction, nrow=n_row)

        batch_stack = torch.stack([input_grid, batch_grid, prediction_grid])
        img_grid = make_grid(batch_stack, nrow=1)

        img_grid = helper.normalize_img_tensor(img_grid)

        if img_grid.max() > 1 or img_grid.min() < 0:
            logging.getLogger().info('Img_grid[{}] value out of range 0-1'.format(batches_done))

        save_image(img_grid.data, "{}/data/Reconstruction_val_{}_{}.png".format(self.logger.log_dir, self.epoch, batches_done), nrow=n_row, normalize=True)
        fig, subplots = helper.create_image_figure(img_grid.cpu(), 'Reconstructed')
        self.logger.add_figure('Impress_AAE_{}_Reconstruction_val_{}_{:0>10}'.format(self.name, self.epoch, batches_done), fig)
        plt.close()