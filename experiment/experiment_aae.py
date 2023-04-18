import os
import datetime
from typing import List, Tuple

import faiss
import numpy as np
import logging
from pathlib import Path
import torch
from torch import nn, Tensor
import torch.nn.init
import torch.utils.data
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import matplotlib as mpl
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from sys import exit

from experiment import helper
from experiment.data_zoo import DataZoo


from utils.utils import TensorboardLogger
from utils.utils import dict_merge
from utils.utils import GPU
from utils.keyboard_menu import KeyboardMenu
import itertools

import experiment.helper

# from models.AE import ConvEncoder, ConvDecoder, Discriminator
from models.AE import ConvEncoder2 as ConvEncoder, ConvDecoder2 as ConvDecoder, Discriminator3 as Discriminator

mpl.use('Agg')


SetupStruct = Tuple[
    List[nn.Module],
    List[nn.Module],
    List[torch.optim.Optimizer],
    List[torch.optim.lr_scheduler._LRScheduler]
]


class Experiment:
    def __init__(self, args):
        self.transform = None
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

            'img_size': 0,
            'channels': 0,

            'optimizer': {
                'lr': 0.0001,
                'step_size': 1,
                'beta1': 0.9,
                'beta2': 0.99,
                'gamma': 0.1
            },

            'model': {
                'latent_size': 256,
            },

            'training': {
                'data': None,

                'batchsize': 1,
                'shuffle': True,
                'worker': 0,
                'sample_interval': None,
                'n_checkpoints': None,
                'grad_vis': False,
            },

            'validation': {
                'data': [None],

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
        self.args = experiment.helper.forward_arguments(self.args, ['batchsize'], ['training', 'validation'])

        self.logger = TensorboardLogger(os.path.join(args['log_dir'], args['name']), modules=[__name__], images_dir=True)

        now = datetime.datetime.now()
        self.args['started_training'] = '%d-%02d-%02d-%02d-%02d' % (now.year, now.month, now.day, now.hour, now.minute)
        self.logger.log_options(self.args, changes)

        self.loader_args = {'num_workers': 8, 'pin_memory': False} if self.args['cuda'] else {}

    def __str__(self) -> str:
        return '{}x{}x{}'\
            .format(self.args['batchsize'], self.args['in'], self.args['out'])

    @staticmethod
    def model(img_shape, latent_size) -> [[nn.Module], [nn.Module]]:
        encoder = ConvEncoder(img_shape, latent_size=latent_size, activation=nn.ReLU, last=nn.LeakyReLU).to(GPU.device)
        decoder = ConvDecoder(img_shape, latent_size=latent_size, activation=nn.ReLU).to(GPU.device)
        discriminator = Discriminator(latent_size=latent_size, activation=nn.ReLU).to(GPU.device)

        pixelwise_loss = torch.nn.MSELoss().to(GPU.device)
        adversarial_loss = torch.nn.BCELoss().to(GPU.device)

        return [[encoder, decoder, discriminator], [pixelwise_loss, adversarial_loss]]

    @staticmethod
    def optimizer(models, lr, step_size, beta1=0.9, beta2=0.99, gamma=0.1):
        optimizer_G = optim.AdamW(itertools.chain(models[0].parameters(), models[1].parameters(), models[2].parameters()), lr=lr, betas=(beta1, beta2))

        scheduler_G = lr_scheduler.StepLR(optimizer_G, step_size=step_size, gamma=gamma)

        return [[optimizer_G], [scheduler_G]]

    @staticmethod
    def get_img_shape(args):
        return (args['channels'], *args['img_size'])

    @staticmethod
    def load_model(path, img_shape, latent_size):
        models = Experiment.model(
            img_shape,
            latent_size,
        )[0]

        models, _, args, _ = helper.load_checkpoint(models, None, args={'resume': path})
        return models

    def setup(self) -> SetupStruct:
        self.transform = transforms.Compose([
            transforms.Resize(self.args['img_size']),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

        # initialize training data
        self.training = Training(**self.args['training'], transform=self.transform, logger=self.logger)
        self.validation = Validation(**self.args['validation'], transform=self.transform, logger=self.logger)

        models, losses = Experiment.model((self.args['channels'], *self.args['img_size']), **self.args['model'])
        optimizers, schedulers = Experiment.optimizer(models, **self.args['optimizer'])

        self.log_model(models, self.args['img_size'], self.args['model']['latent_size'])

        if self.args['resume'] is not None:
            models, optimizers, train_step = experiment.helper.load_checkpoint(models, optimizers)
            Training.global_step = train_step

        return models, losses, optimizers, schedulers

    def log_model(self, models, img_size, latent_size):
        encoder, decoder, discriminator = models
        # self.logger.log_graph(encoder, (torch.rand((1, 1, *img_size), device=GPU.device),))
        # self.logger.log_graph(decoder, (torch.rand((1, latent_size), device=GPU.device),))
        summary(encoder, input_size=(1, *img_size), batch_size=self.args['batchsize'])
        summary(decoder, input_size=(latent_size,), batch_size=self.args['batchsize'])
        summary(discriminator, input_size=(latent_size,), batch_size=self.args['batchsize'])

    @staticmethod
    def accuracy(pred, target):
        return (pred > 0.5).type(torch.cuda.FloatTensor).eq(target).sum().item() / target.shape[0]

    def run(self):
        setup = self.setup()

        start = self.args['start_epoch']
        end = start + self.args['epochs']
        Path(os.path.join(self.logger.log_dir, 'started_trainings_loop.txt')).touch()

        for epoch in range(start, end + 1):
            self.logger.epoch = epoch
            self.training(setup=setup, epoch=epoch)
            self.validation(setup=setup, epoch=self.logger.epoch)


class Training:
    global_step = 0

    def __init__(self, data, batchsize, shuffle, worker, sample_interval, n_checkpoints, logger, grad_vis, **kwargs):
        self.data = DataZoo.get(**data, **kwargs)
        self.data_len = len(self.data)
        self.batch_size = batchsize
        self.batch_count = self.data_len // self.batch_size
        self.dataloader = DataLoader(self.data, batch_size=batchsize, shuffle=shuffle, num_workers=worker)

        self.name = 'Impress_AAE'
        self.logger = logger
        self.sample_interval = sample_interval
        self.n_checkpoints = n_checkpoints
        self.epoch = None
        self.grad_vis = grad_vis
        self.grad_vis_done = not grad_vis
        self.prev_running_loss = 0.0

    def train(self, models, losses, optimizers, imgs, gt_vector):
        # Configure x
        current_batch_size = imgs.shape[0]
        real_imgs = Variable(imgs.type(Tensor).to(GPU.device))
        valid, fake = gt_vector

        encoder, decoder, discriminator = models
        mse_loss, adv_loss = losses
        optimizer, = optimizers

        optimizer.zero_grad()

        # ----------------------
        #  Train Reconstruction
        # ----------------------

        encoded_imgs = encoder(real_imgs)
        x_hat = decoder(encoded_imgs)

        recon_loss = mse_loss(real_imgs, x_hat)

        self.logger.log_value_and_epoch_avg('recon_loss_train', recon_loss.item(), self.epoch, current_batch_size, self.data_len)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Sample noise as discriminator ground truth
        z = Variable(Tensor(np.random.normal(0, 1, (current_batch_size, encoder.latent_size))).to(GPU.device))

        # Measure discriminator's ability to classify real from generated samples
        prediction_real = discriminator(encoded_imgs)
        real_loss = adv_loss(prediction_real, valid[:current_batch_size])
        prediction_fake = discriminator(z)
        fake_loss = adv_loss(prediction_fake, fake[:current_batch_size])
        forging_loss = 0.5 * (real_loss + fake_loss)
        self.logger.log_value_and_epoch_avg('forging_loss_train', forging_loss.item(), self.epoch, current_batch_size, self.data_len)

        adv_weight = 0.1
        loss = (1-adv_weight) * recon_loss + adv_weight * forging_loss
        loss.backward()

        optimizer.step()

        accuracy_valid = Experiment.accuracy(prediction_real.detach(), valid[:current_batch_size])
        accuracy_fake = Experiment.accuracy(prediction_fake.detach(), fake[:current_batch_size])
        accuracy = 0.5 * (accuracy_fake + accuracy_valid)
        self.logger.log_value_and_epoch_avg('accuracy_train', accuracy, self.epoch, current_batch_size, self.data_len)

        return loss

    def __call__(self, setup: SetupStruct, epoch: int):
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
        # Adversarial ground truths
        valid = Variable(Tensor(self.batch_size, 1).to(GPU.device).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(self.batch_size, 1).to(GPU.device).fill_(0.0), requires_grad=False)
        gt_vector = [valid, fake]

        for i, imgs in enumerate(pbar):
            batches_done = epoch * self.batch_count + i
            loss = self.train(models, losses, optimizers, imgs, gt_vector)

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
        forging_loss = self.logger.get_epoch_loss('forging_loss_train', self.data_len)
        recon_loss = self.logger.get_epoch_loss('recon_loss_train', self.data_len)
        accuracy = self.logger.get_epoch_loss('accuracy_train', self.data_len)
        logging.info(f'Training {self.name} epoch {epoch} Loss: {epoch_loss:.6f} ReconLoss: {recon_loss:.6f} ForgingLoss: {forging_loss:.6f} Accuracy: {accuracy:.6f}')

        if not self.grad_vis_done and epoch % 2 == 0:
            model_params_dict = {
                'Encoder': models[0].named_parameters(),
                'Decoder': models[1].named_parameters(),
                'Discriminator': models[2].named_parameters(),
            }
            helper.plot_multiple_grad_flow_lines(model_params_dict, f"{self.logger.log_dir}/data/Grad_flow_{self.epoch}.lines.png")
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
            encoder, decoder, discriminator = models
            z = Tensor(np.random.normal(0, 1, (self.batch_size, decoder.latent_size))).to(GPU.device)
            gen_imgs = decoder(z)
            save_image(gen_imgs.data, f"{self.logger.log_dir}/data/random_sample_{self.epoch}_{batches_done}.png", nrow=n_row // 4 or n_row, normalize=True)

            batch_start = ((batches_done * self.batch_size) % self.data_len)
            batch_end = ((batches_done * self.batch_size + self.batch_size) % self.data_len)

            imgs = self.data[batch_start:batch_end]
            # imgs = [item for sublist in zip([a for a, _ in imgs], [b for _, b in imgs]) for item in sublist]
            imgs = torch.stack(imgs, dim=0)
            imgs = imgs.to(GPU.device)

            batch = encoder(imgs)
            skewed_encoding = batch.detach()
            for i in range(1, self.batch_size):
                skewed_encoding[i] += skewed_encoding[0]
            skewed_images = decoder(skewed_encoding)
            save_image(skewed_images.data, f"{self.logger.log_dir}/data/skewed_sample_{self.epoch}_{batches_done}.png", nrow=n_row // 8 or n_row, normalize=True)
            prediction = discriminator(batch.detach()).detach()
            prediction_bool = (prediction > 0.5).type(torch.cuda.FloatTensor)
            pixels = np.prod(decoder.img_shape)
            prediction_shape = (self.batch_size, decoder.img_shape[0], decoder.img_shape[1] // 2, decoder.img_shape[2])
            prediction = prediction.repeat(1, pixels // 2).view(prediction_shape)
            prediction_bool = prediction_bool.repeat(1, pixels // 2).view(prediction_shape)
            prediction_comb = torch.cat([prediction_bool, prediction], 2)

            batch = decoder(batch)
            input_grid = make_grid(imgs, nrow=n_row // 8 or n_row)
            batch_grid = make_grid(batch.detach(), nrow=n_row // 8 or n_row)
            prediction_grid = make_grid(prediction_comb, nrow=n_row // 8 or n_row)
            batch_stack = torch.stack([input_grid, batch_grid, prediction_grid])

            img_grid = make_grid(batch_stack, nrow=1)
            img_grid = helper.normalize(img_grid)
            save_image(img_grid.data, f"{self.logger.log_dir}/data/Reconstruction_{self.epoch}_{batches_done}.png", nrow=n_row)
        [model.train() for model in models]


class Validation:
    global_step = 0

    def __init__(self, data, batchsize, worker, logger, sample_interval, save_best_model=True, **kwargs):
        self.data = DataZoo.get(**data, **kwargs)
        self.batch_size = batchsize
        self.dataloader = DataLoader(self.data, batch_size=batchsize, shuffle=False, num_workers=worker)
        self.name = 'Impress_AE'
        self.logger = logger
        self.sample_interval = sample_interval
        self.epoch = None
        self.save_best_model = save_best_model
        self.best_loss = float('inf')
        self.data_len = len(self.data)
        self.batch_count = self.data_len // self.batch_size

    def train(self, models, losses, data, gt_vector):
        encoder, decoder, discriminator = models
        mse_loss, adv_loss = losses
        valid, fake = gt_vector
        labels, imgs = data

        # Configure x
        real_imgs = Variable(imgs.type(Tensor).to(GPU.device))
        current_batch_size = imgs.shape[0]

        # -------------------------
        #  Validate Reconstruction
        # -------------------------

        encoded_imgs = encoder(real_imgs)
        x_hat = decoder(encoded_imgs)
        self.logger.accumulate_embedding_set_for_epoch(encoded_imgs.cpu(), labels, images=imgs, name=self.name)

        recon_loss = mse_loss(real_imgs, x_hat)
        self.logger.log_value_and_epoch_avg('recon_loss_val', recon_loss.item(), self.epoch, self.batch_size, self.data_len)

        # ---------------------
        #  Validate Discriminator
        # ---------------------

        # Sample noise as discriminator ground truth
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], encoder.latent_size))).to(GPU.device))

        # Measure discriminator's ability to classify real from generated samples
        prediction_real = discriminator(encoded_imgs)
        real_loss = adv_loss(prediction_real, valid[:current_batch_size])
        prediction_fake = discriminator(z)
        fake_loss = adv_loss(prediction_fake, fake[:current_batch_size])
        forging_loss = 0.5 * (real_loss + fake_loss)
        self.logger.log_value_and_epoch_avg('forging_loss_val', forging_loss.item(), self.epoch, self.batch_size, self.data_len)

        accuracy_valid = Experiment.accuracy(prediction_real.detach(), valid[:current_batch_size])
        accuracy_fake = Experiment.accuracy(prediction_fake.detach(), fake[:current_batch_size])
        accuracy = 0.5 * (accuracy_fake + accuracy_valid)
        self.logger.log_value_and_epoch_avg('accuracy_val', accuracy, self.epoch, self.batch_size, self.data_len)
        adv_weight = 0.1
        loss = (1 - adv_weight) * recon_loss + adv_weight * forging_loss
        return loss

    def __call__(self, setup, epoch):
        self.epoch = epoch
        models, losses, optimizers, _ = setup
        [model.eval() for model in models]

        pbar = tqdm(self.dataloader)
        pbar.set_description('Validation Epoch {}'.format(epoch))
        scaled_sample_interval = self.sample_interval * self.batch_count
        # Adversarial ground truths
        valid = Variable(Tensor(self.batch_size, 1).to(GPU.device).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(self.batch_size, 1).to(GPU.device).fill_(0.0), requires_grad=False)
        gt_vector = [valid, fake]

        for i, imgs in enumerate(pbar):
            batches_done = epoch * self.batch_count + i
            loss = self.train(models, losses, imgs, gt_vector)

            if self.sample_interval is not None and batches_done % scaled_sample_interval == 0:
                self.sample_image(models, n_row=self.batch_size, batches_done=batches_done)

            # log values
            self.logger.step(self.batch_size)
            Validation.global_step += self.batch_size

            # calculate log values
            self.logger.log_value_and_epoch_avg('validation_loss', loss.item(), self.epoch, self.batch_size, self.data_len)

        epoch_loss = self.logger.get_epoch_loss('validation_loss', self.data_len)
        epoch_accuracy = self.logger.get_epoch_loss('validation_loss', self.data_len)

        kmeans_acc = 0.0
        if epoch_loss < self.best_loss:
            self.best_loss = epoch_loss
            helper.save_checkpoint(epoch, Training.global_step, models, optimizers, self.logger.log_dir, best=True)
            features, labels, labels_header, images = self.logger.get_embedding_set(self.name)
            clusters = len(labels.unique())
            fmeans = faiss.Kmeans(features.shape[1], clusters)
            fmeans.train(features.detach().numpy())
            cluster_centers, cluster_ids_x = fmeans.assign(features.detach().numpy())
            for n in range(1, clusters + 1, 2):
                kmeans_acc += 2 if cluster_ids_x[n - 1] == cluster_ids_x[n] else 0
            kmeans_acc /= clusters + 1
            self.logger.log_value('Kmeans_acc_val', kmeans_acc)

            self.logger.log_embedding_set(self.name, step=epoch)
        else:
            self.logger.clear_embedding_set(self.name)

        logging.info(f'Validation {self.name} epoch {epoch} Loss: {epoch_loss:.6f} Accuracy: {epoch_accuracy:.6f} ClusterAccuracy: {kmeans_acc:.6f}')

        # do checkpointing
        return  # do training

    def sample_image(self, models, n_row, batches_done):
        """Saves a grid of generated data"""
        with torch.no_grad():
            encoder, decoder, discriminator = models
            z = Tensor(np.random.normal(0, 1, (self.batch_size, decoder.latent_size))).to(GPU.device)
            gen_imgs = decoder(z)
            save_image(gen_imgs.data, f"{self.logger.log_dir}/data/random_sample_val_{self.epoch}_{batches_done}.png", nrow=n_row // 8 or n_row, normalize=True)

            batch_start = ((batches_done * self.batch_size) % self.data_len)
            batch_end = ((batches_done * self.batch_size + self.batch_size) % self.data_len)

            imgs = self.data[batch_start:batch_end]
            # imgs = [item for sublist in zip([a for a, _ in imgs], [b for _, b in imgs]) for item in sublist]
            imgs = [img for _, img in imgs]
            imgs = torch.stack(imgs, dim=0)
            imgs = imgs.to(GPU.device)

            batch = encoder(imgs)
            skewed_encoding = batch.detach()
            for i in range(1, self.batch_size):
                skewed_encoding[i] += skewed_encoding[0]
            skewed_images = decoder(skewed_encoding)
            save_image(skewed_images.data, f"{self.logger.log_dir}/data/skewed_sample_val_{self.epoch}_{batches_done}.png", nrow=n_row // 8 or n_row, normalize=True)
            prediction = discriminator(batch.detach()).detach()
            prediction_bool = (prediction > 0.5).type(torch.cuda.FloatTensor)
            pixels = np.prod(decoder.img_shape)
            prediction_shape = (self.batch_size, decoder.img_shape[0], decoder.img_shape[1] // 2, decoder.img_shape[2])
            prediction = prediction.repeat(1, pixels // 2).view(prediction_shape)
            prediction_bool = prediction_bool.repeat(1, pixels // 2).view(prediction_shape)
            prediction_comb = torch.cat([prediction_bool, prediction], 2)

            batch = decoder(batch)
            input_grid = make_grid(imgs, nrow=n_row // 8 or n_row)
            batch_grid = make_grid(batch.detach(), nrow=n_row // 8 or n_row)
            prediction_grid = make_grid(prediction_comb, nrow=n_row // 8 or n_row)
            batch_stack = torch.stack([input_grid, batch_grid, prediction_grid])

            img_grid = make_grid(batch_stack, nrow=1)
            img_grid = helper.normalize(img_grid)
            save_image(img_grid.data, f"{self.logger.log_dir}/data/Reconstruction_val_{self.epoch}_{batches_done}.png", nrow=n_row // 8 or n_row)