import os
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
import itertools

import experiment.helper as helper
from torchsummary import summary

from models.SupRVAE import ConvEncoder, ConvDecoder, Discriminator
# from models.SupRVAE import ConvDirEncoder as ConvEncoder, ConvDirDecoder as ConvDecoder, Discriminator
# from models.SupRVAE import ConvResEncoder as ConvEncoder, ConvResDecoder as ConvDecoder, Discriminator
# from models.SupRVAE import ConvParEncoder as ConvEncoder, ConvParDecoder as ConvDecoder, Discriminator
from models.utils import Mish, Swish, Sin

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
                'encoder_depth': 3,
                'decoder_depth': 3,
                'discriminator_depth': 3,
                'residual_enc': True,
                'residual_dec': True,
                'start_filter_enc': 16,
                'start_pixels_dec': 32,
            },

            'img_size': 0,
            'channels': 0,

            'optimizer': {
                'lr': 0.001,
                'step_size': 50,
                'beta1': 0.99,
                'beta2': 0.999,
                'gamma': 0.5
            },

            'training': {
                'data': None,

                'batchsize': 1,
                'shuffle': True,
                'worker': 0,
                'sample_interval': None,
                'sample_latentspace': True,
                'n_checkpoints': None,
                'grad_vis': False
            },

            'validation': {
                'data': [None],
                'sample_interval': None,
                'sample_latentspace': True,
                'n_checkpoints': None,
                'grad_vis': False,

                'batchsize': 1,
                'worker': 0,
                'plot': None,
            }
        }

        self.keyboard_menu = None
        self.transform = None

        # check config by comparing with defaults and merge
        changes = dict_merge(self.args, args, verify=True)

        # forward arguments
        self.args = helper.forward_arguments(self.args, ['batchsize'], ['training'])

        self.logger = TensorboardLogger(os.path.join(args['log_dir'], args['name']), modules=[__name__], images_dir=True)

        now = datetime.datetime.now()
        self.args['started_training'] = '%d-%02d-%02d-%02d-%02d' % (now.year, now.month, now.day, now.hour, now.minute)
        self.logger.log_options(self.args, changes)

        self.loader_args = {'num_workers': 8, 'pin_memory': False} if self.args['cuda'] else {}

    def __str__(self) -> str:
        return '{}x{}x{}'\
            .format(self.args['batchsize'], self.args['in'], self.args['out'])

    @staticmethod
    def model(
            batch_size,
            img_shape,
            latent_size,
            encoder_depth,
            decoder_depth,
            discriminator_depth,
            residual_enc,
            residual_dec,
            start_pixels_dec,
            start_filter_enc,
            show_summary=True
    ) -> [[nn.Module], [nn.Module]]:
        activation = nn.ReLU
        encoder_last = nn.ReLU
        decoder_last = nn.Sigmoid
        encoder = ConvEncoder(
            img_shape, latent_size=latent_size, depth=encoder_depth, activation=activation,
            residual=residual_enc, start_filter=start_filter_enc, last=encoder_last
            # , conv_block_length=3
        ).to(GPU.device)
        decoder = ConvDecoder(
            img_shape, latent_size=latent_size, depth=decoder_depth, activation=activation,
            residual=residual_dec, start_pixels=start_pixels_dec, last=decoder_last
            # , conv_block_length=3
        ).to(GPU.device)
        discriminator = Discriminator(latent_size, depth=discriminator_depth, activation=activation, dropout=0.2).to(GPU.device)

        if show_summary:
            summary(encoder, input_size=img_shape, batch_size=batch_size)
            summary(decoder, input_size=(latent_size,), batch_size=batch_size)
            summary(discriminator, input_size=(latent_size,), batch_size=batch_size)

        adversarial_loss = torch.nn.BCELoss().to(GPU.device)
        pixelwise_loss = torch.nn.BCELoss().to(GPU.device)
        discriminator_loss = torch.nn.BCELoss().to(GPU.device)

        return [[encoder, decoder, discriminator], [adversarial_loss, pixelwise_loss, discriminator_loss]]

    @staticmethod
    def optimizer(models, lr, step_size, beta1=0.9, beta2=0.99, gamma=0.1):
        encoder, decoder, discriminator = models
        optimizer_G = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

        scheduler_G = lr_scheduler.StepLR(optimizer_G, step_size=step_size, gamma=gamma)
        scheduler_D = lr_scheduler.StepLR(optimizer_D, step_size=step_size, gamma=gamma)

        return [[optimizer_G, optimizer_D], [scheduler_G, scheduler_D]]

    @staticmethod
    def get_img_shape(args, padding=0):
        if not isinstance(args['img_size'], list):
            return args['channels'], args['img_size'] + padding, args['img_size'] + padding
        else:
            return args['channels'], args['img_size'][0] + padding, args['img_size'][1] + padding

    @staticmethod
    def load_model(path, img_shape, latent_size, depth, start_filter, start_pixels, residual=True, show_summary=True):
        img_shape = (1, img_shape, img_shape)
        models = Experiment.model(
            1,
            img_shape,
            latent_size,
            encoder_depth=depth,
            decoder_depth=depth,
            discriminator_depth=depth,
            residual_enc=residual,
            residual_dec=residual,
            start_filter_enc=start_filter,
            start_pixels_dec=start_pixels,
            show_summary=show_summary
        )[0]

        models, _, args = helper.load_checkpoint(models, None, args={'resume': path})
        return models

    def setup(self) -> SetupStruct:
        img_shape = self.get_img_shape(self.args)

        self.transform = transforms.Compose([
            # transforms.Resize(img_shape[1:]),
            transforms.Grayscale(),
            # transforms.RandomChoice([
            #     transforms.RandomHorizontalFlip(),
            #     transforms.RandomResizedCrop(img_shape[1], (0.8, 1.0)),
            #     transforms.RandomAffine(30, fillcolor=255),
            #     transforms.RandomAffine(0, (0.1, 0.1), fillcolor=255),
            #     transforms.Compose([
            #         transforms.RandomHorizontalFlip(),
            #         transforms.RandomResizedCrop(img_shape[1], (0.8, 1.0)),
            #     ]),
            #     transforms.Compose([
            #         transforms.RandomAffine(30, fillcolor=255),
            #         transforms.RandomAffine(0, (0.1, 0.1), fillcolor=255),
            #     ])
            # ]),
            transforms.ToTensor(),
        ])

        # initialize training data
        self.training = Training(**self.args['training'], transform=self.transform, logger=self.logger)
        self.validate = Validate(**self.args['training'], transform=self.transform, logger=self.logger)

        models, losses = Experiment.model(self.args['batchsize'], img_shape, **self.args['model'])
        optimizers, schedulers = Experiment.optimizer(models, **self.args['optimizer'])

        if self.args['resume'] is not None:
            models, optimizers, args, train_step = helper.load_checkpoint(models, optimizers, self.args)
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
            self.validate(setup=setup, epoch=epoch, menu=self.keyboard_menu)

        helper.save_checkpoint(end + 1, Training.global_step, setup[0], setup[2], self.logger.log_dir, self.args['training']['n_checkpoints'])


class Training:
    global_step = 0

    def __init__(self, data, batchsize, shuffle, worker, sample_interval, n_checkpoints, logger, grad_vis, sample_latentspace, **kwargs):
        self.data = DataZoo.get(**data, **kwargs)
        self.dataloader = DataLoader(self.data, batch_size=batchsize, shuffle=shuffle, num_workers=worker)
        self.data_len = len(self.data)
        self.batch_size = batchsize

        self.logger = logger
        self.sample_interval = sample_interval
        self.n_checkpoints = n_checkpoints
        self.epoch = None
        self.grad_vis = grad_vis
        self.grad_vis_done = not grad_vis
        self.sample_latentspace = sample_latentspace

    def train_g(self, models: List[nn.Module], losses: List[nn.Module], optimizers, input, gt_vec, batches_done):
        encoder, decoder, discriminator = models
        adversarial_loss, pixelwise_loss, _ = losses
        optimizer_G = optimizers[0]
        valid, fake = gt_vec
        # adv_weight = (torch.sigmoid(torch.Tensor(1).fill_(self.epoch)) - 0.5).item()
        adv_weight = 0.5
        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        encoded_imgs = encoder(input)
        decoded_imgs = decoder(encoded_imgs)

        z = torch.Tensor(np.random.normal(0, 1, (encoded_imgs.shape[0], encoder.latent_size))).to(GPU.device)

        # Loss measures generator's ability to fool the discriminator
        g_forging_loss = adversarial_loss(discriminator(encoder(decoder(z))), fake[:encoded_imgs.shape[0]])
        # logging.info('Forging loss {:.10f}'.format(g_forging_loss))
        g_reconstruction_loss = pixelwise_loss(decoded_imgs, input) + adversarial_loss(discriminator(encoded_imgs), valid[:encoded_imgs.shape[0]])
        # logging.info('Reconstruction loss {:.10f}'.format(g_reconstruction_loss))

        # logging.info('Reconstruction loss {:.10f}'.format(g_reconstruction_loss))
        self.logger.log_value_and_epoch_avg('train_loss_g_forge', g_forging_loss.item(), self.epoch, self.batch_size, self.data_len)
        self.logger.log_value_and_epoch_avg('train_loss_g_recon', g_reconstruction_loss.item(), self.epoch, self.batch_size, self.data_len)

        g_loss = adv_weight * g_forging_loss + (1 - adv_weight) * g_reconstruction_loss
        # logging.info('Combined loss {:.10f}'.format(g_loss))
        self.logger.log_value_and_epoch_avg('train_loss_g', g_loss.item(), self.epoch, self.batch_size, self.data_len)

        g_loss.backward()
        optimizer_G.step()

        return g_loss, encoded_imgs

    def accuracy(self, pred, target):
        return (pred > 0.5).type(torch.cuda.FloatTensor).eq(target).sum().item() / pred.shape[0]

    def train_d(self, models: List[nn.Module], losses: List[nn.Module], optimizers, input, gt_vec: List[torch.autograd.Variable], batches_done):
        encoder, decoder, discriminator = models
        discriminator_loss, = losses[2:]
        optimizer_D = optimizers[1]
        valid, fake = gt_vec
        real_imgs, encoded_imgs = input

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as discriminator ground truth
        current_batch_size = real_imgs.shape[0]
        z = torch.Tensor(np.random.normal(0, 1, (current_batch_size, encoder.latent_size))).to(GPU.device)

        # Measure discriminator's ability to classify real from generated samples
        pred = discriminator(z)

        # Calculate accuracy for fake distributions
        accuracy_fake = self.accuracy(pred, fake[:current_batch_size])
        fake_loss = discriminator_loss(pred, fake)
        self.logger.log_value_and_epoch_avg('train_loss_d_fake', fake_loss.item(), self.epoch, current_batch_size, self.data_len)
        self.logger.log_value_and_epoch_avg('train_acc_d_fake', accuracy_fake, self.epoch, current_batch_size, self.data_len)

        pred = discriminator(encoded_imgs.detach())

        # Calculate accuracy for real distributions
        accuracy_valid = self.accuracy(pred, valid[:current_batch_size])
        real_loss = discriminator_loss(pred, valid)
        self.logger.log_value_and_epoch_avg('train_loss_d_real', real_loss.item(), self.epoch, current_batch_size, self.data_len)
        self.logger.log_value_and_epoch_avg('train_acc_d_real', accuracy_valid, self.epoch, current_batch_size, self.data_len)

        d_loss = 0.5 * (real_loss + fake_loss)
        accuracy = 0.5 * (accuracy_fake + accuracy_valid)
        self.logger.log_value_and_epoch_avg('train_loss_d', d_loss.item(), self.epoch, current_batch_size, self.data_len)
        self.logger.log_value_and_epoch_avg('train_acc_d', accuracy, self.epoch, current_batch_size, self.data_len)

        d_loss.backward()
        optimizer_D.step()

        return d_loss, accuracy

    def train(self, models, losses, optimizers, imgs, gt_vec, batches_done):
        # Configure x
        real_imgs = imgs.type(Tensor).to(GPU.device)

        g_loss, encoded_imgs = self.train_g(models, losses, optimizers, real_imgs, gt_vec, batches_done)

        d_loss, accuracy = self.train_d(models, losses, optimizers, [real_imgs, encoded_imgs], gt_vec, batches_done)

        return g_loss, d_loss, accuracy

    def __call__(self, setup: SetupStruct, epoch: int, menu):
        models, losses, optimizers, schedulers = setup
        [model.train() for model in models]
        scheduler_g, scheduler_d = schedulers
        # scheduler_g.step(); scheduler_d.step()

        self.epoch = epoch
        self.grad_vis_done = not self.grad_vis

        self.logger.log_value('lr_g', scheduler_g.get_last_lr()[0], Training.global_step)
        self.logger.log_value('lr_d', scheduler_d.get_last_lr()[0], Training.global_step)

        running_loss_D = 0.0
        running_loss_G = 0.0
        running_acc_D = 0.0

        pbar = tqdm(self.dataloader)
        pbar.set_description('Epoch {}'.format(epoch))

        # Adversarial ground truths
        valid = torch.Tensor(self.dataloader.batch_size, 1).to(GPU.device).fill_(1.0).requires_grad_(False)
        fake = torch.Tensor(self.dataloader.batch_size, 1).to(GPU.device).fill_(0.0).requires_grad_(False)

        for i, imgs in enumerate(pbar):
            batches_done = epoch * self.data_len + i
            g_loss, d_loss, accuracy = self.train(models, losses, optimizers, imgs, [valid, fake], batches_done)

            if not self.grad_vis_done and epoch % 2 == 0:
                model_params_dict = {
                    'SuperVAE_Encoder': models[0].named_parameters(),
                    'SuperVAE_Decoder': models[1].named_parameters(),
                }
                helper.plot_multiple_grad_flow_lines(model_params_dict, "{}/data/Grad_flow_{}.lines.png".format(self.logger.log_dir, self.epoch))
                self.grad_vis_done = True

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
        epoch_loss_G = running_loss_G / self.data_len
        epoch_loss_D = running_loss_D / self.data_len
        epoch_acc_D = running_acc_D / self.data_len
        # adv_weight = (torch.sigmoid(torch.Tensor(1).fill_(epoch)) - 0.5).item()
        adv_weight = 0.5
        self.logger.log_value('train_epoch_g_loss', epoch_loss_G, Training.global_step)
        self.logger.log_value('train_epoch_d_loss', epoch_loss_D, Training.global_step)
        self.logger.log_value('train_epoch_d_acc', epoch_acc_D, Training.global_step)
        self.logger.log_value('train_epoch_adv_weight', adv_weight, Training.global_step)
        logging.info('Training {} epoch {} Loss_G: {:.4f} Loss_D: {:.4f} Accuracy_D: {:.4} Lr_g: {:.4} Lr_d: {:.4} Adv-Weight: {:.4}'
                     .format('Impress_SVRAE', epoch, epoch_loss_G, epoch_loss_D, epoch_acc_D, scheduler_g.get_lr()[0], scheduler_d.get_lr()[0], adv_weight))

        # do checkpointing
        if self.n_checkpoints is None or epoch % self.n_checkpoints == 0:
            helper.save_checkpoint(epoch, Training.global_step, models, optimizers, self.logger.log_dir, self.n_checkpoints)
        [scheduler.step() for scheduler in schedulers]

        return  # do training

    def sample_image(self, models, n_row, batches_done):
        """Saves a grid of generated data"""
        # [model.eval() for model in models]
        encoder, decoder, discriminator = models

        if self.sample_latentspace:
            # Sample noise
            z = Tensor(np.random.normal(0, 1, (n_row, decoder.latent_size))).to(GPU.device).requires_grad_(False)
            gen_imgs = decoder(z)
            gen_imgs = helper.normalize(gen_imgs)

            save_image(gen_imgs.data, "{}/data/sample_e{:0>3}_{}.png".format(self.logger.log_dir, self.epoch, batches_done), nrow=n_row, normalize=True)
            img_grid = make_grid(gen_imgs.detach(), nrow=n_row)
            fig, subplots = helper.create_image_figure(img_grid.cpu(), 'Generated', 1, 1)
            self.logger.add_figure('{}_Sample_E{:0>3}_{:0>10}'.format('Impress_SupRVAE', self.epoch, batches_done), fig)
            plt.close()

        batch_size = self.dataloader.batch_size
        batch_start = ((batches_done * batch_size) % len(self.data))
        batch_end = ((batches_done * batch_size + batch_size) % len(self.data))
        imgs = self.data[batch_start:batch_end]
        imgs = torch.stack(imgs)
        imgs = imgs.to(GPU.device)

        # self.logger.log_graph(encoder, imgs.detach())

        batch = encoder(imgs.detach()).detach()
        # self.logger.log_graph(decoder, batch.detach())

        prediction = discriminator(batch.detach()).detach()
        pixels = np.prod(decoder.img_shape)
        prediction = prediction.repeat(1, pixels).view((batch_size, *decoder.img_shape))
        batch = decoder(batch)
        batch_grid = make_grid(batch.detach(), nrow=n_row)
        input_grid = make_grid(imgs, nrow=n_row)
        prediction_grid = make_grid(prediction, nrow=n_row)

        batch_stack = torch.stack([input_grid, batch_grid, prediction_grid])
        img_grid = make_grid(batch_stack, nrow=1)
        img_grid = helper.normalize(img_grid)
        save_image(img_grid.data, "{}/data/Reconstruction_E{:0>3}_{}.png".format(self.logger.log_dir, self.epoch, batches_done), nrow=n_row, normalize=True)
        # helper.save_image(img_grid, "{}/data/Reconstruction_{}.pil".format(self.logger.log_dir, batches_done))
        fig, subplots = helper.create_image_figure(img_grid.cpu(), 'Reconstructed')
        self.logger.add_figure('{}_Reconstruction_E{:0>3}__{:0>10}'.format('Impress_SupVAE', self.epoch, batches_done), fig)
        plt.close()
        # fig.savefig("{}/data/Reconstruction_fig_{:0>10}.png".format(self.logger.log_dir, batches_done))


class Validate:
    global_step = 0

    def __init__(self, data, batchsize, shuffle, worker, sample_interval, n_checkpoints, logger, grad_vis, sample_latentspace, **kwargs):
        self.data = DataZoo.get(**data, **kwargs)
        self.dataloader = DataLoader(self.data, batch_size=batchsize, shuffle=shuffle, num_workers=worker)
        self.batch_size = batchsize
        self.data_len = len(self.data)

        self.logger = logger
        self.sample_interval = sample_interval
        self.n_checkpoints = n_checkpoints
        self.grad_vis = grad_vis
        self.sample_latentspace = sample_latentspace
        self.best_loss = float('inf')

    def train_g(self, models: List[nn.Module], losses: List[nn.Module], optimizers, input, gt_vec):
        encoder, decoder, discriminator = models
        adversarial_loss, pixelwise_loss, _ = losses
        optimizer_G = optimizers[0]
        valid, fake = gt_vec
        adv_weight = 0.5
        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        encoded_imgs = encoder(input)
        decoded_imgs = decoder(encoded_imgs)

        z = torch.Tensor(np.random.normal(0, 1, (self.dataloader.batch_size, encoder.latent_size))).to(GPU.device)

        # Loss measures generator's ability to fool the discriminator
        g_forging_loss = adversarial_loss(discriminator(encoder(decoder(z))), fake)
        g_reconstruction_loss = pixelwise_loss(decoded_imgs, input) + adversarial_loss(discriminator(encoded_imgs), valid)
        self.logger.log_value_and_epoch_avg('validation_loss_g_forge', g_forging_loss.item(), self.epoch, self.batch_size, self.data_len)
        self.logger.log_value_and_epoch_avg('validation_loss_g_recon', g_reconstruction_loss.item(), self.epoch, self.batch_size, self.data_len)

        g_loss = adv_weight * g_forging_loss + (1 - adv_weight) * g_reconstruction_loss
        self.logger.log_value_and_epoch_avg('validation_loss_g', g_loss.item(), self.epoch, self.batch_size, self.data_len)

        g_loss.backward()

        optimizer_G.step()

        return g_loss, encoded_imgs

    def accuracy(self, pred, target):
        return (pred > 0.5).type(torch.cuda.FloatTensor).eq(target).sum().item() / self.dataloader.batch_size

    def train_d(self, models: List[nn.Module], losses: List[nn.Module], optimizers, input, gt_vec: List[torch.autograd.Variable]):
        encoder, decoder, discriminator = models
        discriminator_loss, = losses[2:]
        optimizer_D = optimizers[1]
        valid, fake = gt_vec
        real_imgs, encoded_imgs = input

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as discriminator ground truth
        z = torch.Tensor(np.random.normal(0, 1, (self.dataloader.batch_size, encoder.latent_size))).to(GPU.device)

        # Measure discriminator's ability to classify real from generated samples
        pred = discriminator(z)

        # Calculate accuracy for fake distributions
        accuracy_fake = self.accuracy(pred, fake)
        fake_loss = discriminator_loss(pred, fake)
        self.logger.log_value_and_epoch_avg('validation_loss_d_fake', fake_loss.item(), self.epoch, self.batch_size, self.data_len)
        self.logger.log_value_and_epoch_avg('validation_accuracy_d_fake', accuracy_fake, self.epoch, self.batch_size, self.data_len)

        pred = discriminator(encoded_imgs.detach())

        # Calculate accuracy for real distributions
        accuracy_valid = self.accuracy(pred, valid)
        real_loss = discriminator_loss(pred, valid)
        self.logger.log_value_and_epoch_avg('validation_loss_d_real', real_loss.item(), self.epoch, self.batch_size, self.data_len)
        self.logger.log_value_and_epoch_avg('validation_accuracy_d_real', accuracy_valid, self.epoch, self.batch_size, self.data_len)

        d_loss = 0.5 * (real_loss + fake_loss)
        accuracy = 0.5 * (accuracy_fake + accuracy_valid)
        self.logger.log_value_and_epoch_avg('validation_loss_d', d_loss.item(), self.epoch, self.batch_size, self.data_len)
        self.logger.log_value_and_epoch_avg('validation_accuracy_d', accuracy, self.epoch, self.batch_size, self.data_len)

        d_loss.backward()
        optimizer_D.step()

        return d_loss, accuracy

    def eval(self, models, losses, optimizers, imgs, gt_vec):
        # Configure x
        real_imgs = Variable(imgs.type(Tensor).to(GPU.device))

        g_loss, encoded_imgs = self.train_g(models, losses, optimizers, real_imgs, gt_vec)

        d_loss, accuracy = self.train_d(models, losses, optimizers, [real_imgs, encoded_imgs], gt_vec)

        return g_loss, d_loss, accuracy

    def __call__(self, setup: SetupStruct, epoch: int, menu):
        self.epoch = epoch
        models, losses, optimizers, _ = setup
        [model.eval() for model in models]
        encoder, _, _ = models

        pbar = tqdm(self.dataloader)
        pbar.set_description('Epoch {}'.format(epoch))
        gt_vec = [Variable(torch.ones(self.dataloader.batch_size).type(Tensor).to(GPU.device)), Variable(torch.zeros(self.dataloader.batch_size).type(Tensor).to(GPU.device))]
        for i, imgs in enumerate(pbar):
            real_imgs = Variable(imgs.type(Tensor).to(GPU.device))
            g_loss, d_loss, accuracy = self.eval(models, losses, optimizers, real_imgs, gt_vec)
            loss = g_loss + d_loss
            batches_done = epoch * len(self.data) + i
            if batches_done % self.sample_interval == 0:
                self.sample_image(models, n_row=self.dataloader.batch_size, batches_done=batches_done)

            # log values
            self.logger.step(self.dataloader.batch_size)
            Validate.global_step += self.dataloader.batch_size
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
        return  # do training

    def sample_image(self, models, n_row, batches_done):
        """Saves a grid of generated data"""
        encoder, decoder, discriminator = models
        # Sample noise
        z = Variable(Tensor(np.random.normal(0, 1, (n_row, decoder.latent_size))).to(GPU.device))
        gen_imgs = decoder(z)

        save_image(gen_imgs.data, "{}/data/sample_val_E{:0>3}_{}.val.png".format(self.logger.log_dir, self.epoch, batches_done), nrow=n_row, normalize=True)
        img_grid = make_grid(gen_imgs.detach(), nrow=n_row, normalize=True)
        fig, subplots = helper.create_image_figure(img_grid.cpu(), 'Generated', 1, 1)
        self.logger.add_figure('{}_Sample_val_E{:0>3}_{:0>10}'.format('Impress_SupVAE', self.epoch, batches_done), fig)

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
        save_image(img_grid.data, "{}/data/Reconstruction_val_E{:0>3}_{}.png".format(self.logger.log_dir, self.epoch, batches_done), nrow=n_row, normalize=True)
        fig, subplots = helper.create_image_figure(img_grid.cpu(), 'Reconstructed')
        self.logger.add_figure('{}_Reconstruction_val_E{:0>3}_{:0>10}'.format('Impress_SupVAE', self.epoch, batches_done), fig)