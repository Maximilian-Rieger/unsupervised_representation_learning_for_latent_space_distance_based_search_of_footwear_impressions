import os
import asyncio
import datetime
from typing import Tuple, List

import einops
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
import itertools

from models.experimental.perceiver_2_vae import PerceiverVLT
from models.experimental.perceiver_2 import Perceiver, Deceiver, Discriminator
from optimizers.AdaBelief import AdaBelief

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
        res = self.wrapped(x)
        return res[self.forward_pos]


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
            'scatter_plot_interval': None,
            'img_size': (256, 256),

            'model': {
                'num_freq_bands': 6,
                'encoder_depth': 6,
                'decoder_depth': 3,
                'max_freq': 64,
                'freq_base': 2,
                'input_channels': 3,
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
                'lr': (0.01, 0.001),
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
                'sample_interval': None,
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
        return f'{self.args["batchsize"]}x{self.args["in"]}x{self.args["out"]}'

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
            show_summary=True
    ) -> [[nn.Module], [nn.Module]]:
        perceiver = PerceiverVLT(
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

        discriminator = Discriminator(latent_dim=latent_dim).to(GPU.device)

        if show_summary:
            summary(ApplyWrapper(perceiver, 0), input_size=(64, 64, 1), batch_size=batch_size)
            summary(FuncApplyWrapper(deceiver, lambda _: [[2, 64, 64, 1]]), input_size=(latent_dim,), batch_size=batch_size)
            summary(discriminator, input_size=(latent_dim,), batch_size=batch_size)

        recon_loss = torch.nn.MSELoss(reduction='mean').to(GPU.device)
        adversarial_loss = torch.nn.BCELoss(reduction='mean').to(GPU.device)

        return [[perceiver, deceiver, discriminator], [recon_loss, adversarial_loss]]

    @staticmethod
    def load_model(path, *args, **kwargs):
        models = Experiment.model(*args, **kwargs, show_summary=False)[0]
        models, _, args, _ = helper.load_checkpoint(models, None, args={'resume': path})
        return models

    @staticmethod
    def optimizer(models, lr, step_size, beta1=0.9, beta2=0.99, gamma=0.1):
        perceiver, deceiver, discriminator = models

        lr_g, lr_d = lr

        optimizer = torch.optim.AdamW(itertools.chain(perceiver.parameters(), deceiver.parameters()), lr=lr_g, betas=(beta1, beta2))
        optimizer_D = torch.optim.AdamW(itertools.chain(perceiver.parameters(), discriminator.parameters()), lr=lr_d, betas=(beta1, beta2))

        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        scheduler_D = lr_scheduler.StepLR(optimizer_D, step_size=step_size, gamma=gamma)

        return [[optimizer, optimizer_D], [scheduler, scheduler_D]]

    def setup(self) -> SetupStruct:
        self.transform = transforms.Compose([
            transforms.Resize(self.args['img_size']),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

        # initialize training data
        self.training = Training(**self.args['training'], transform=self.transform, logger=self.logger)
        self.validate = Validation(**self.args['validation'], transform=self.transform, logger=self.logger)

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
            self.training(setup=setup, epoch=epoch)
            del self.training.encoder_grads
            self.validate(setup=setup, epoch=epoch)

        helper.save_checkpoint(end + 1, Training.global_step, setup[0], setup[2], self.logger.log_dir, self.args['training']['n_checkpoints'])

    @staticmethod
    def accuracy(pred, target, batch_size):
        return (pred.detach() > 0.5).type(torch.cuda.FloatTensor).eq(target.detach()).sum().item() / batch_size


class Training:
    global_step = 0

    def __init__(self, data, batchsize, shuffle, worker, sample_interval, n_checkpoints, logger, grad_vis, **kwargs):
        self.data = DataZoo.get(**data, **kwargs)
        self.dataloader = DataLoader(self.data, batch_size=batchsize, shuffle=shuffle, num_workers=worker)

        self.name = 'Impress_Perceiver_AT'
        self.logger = logger
        self.sample_interval = sample_interval
        self.n_checkpoints = n_checkpoints
        self.epoch = None
        self.encoder_grads = None
        self.grad_vis = grad_vis
        self.grad_vis_done = not grad_vis
        self.batch_size = batchsize
        self.data_len = len(self.data)
        self.batch_count = self.data_len // self.batch_size

    def train(self, models, losses, optimizers, imgs, gt):
        # Configure x
        real_imgs = Variable(imgs.type(Tensor).to(GPU.device))

        input_imgs = rearrange(real_imgs, 'b c h w -> b h w c')
        img_shape = input_imgs.shape

        perceiver, deceiver, discriminator = models
        mse_loss, adversarial_loss = losses
        optimizer_recon, optimizer_regul = optimizers
        valid, fake = gt

        current_bs = self.dataloader.batch_size
        if real_imgs.shape[0] != self.dataloader.batch_size:
            current_bs = real_imgs.shape[0]
            valid = valid[0:current_bs, :]
            fake = fake[0:current_bs, :]

        # -------------------------------
        #     Reconstruction Training
        # -------------------------------
        optimizer_recon.zero_grad()

        # mask = Tensor(np.random.normal(0, 1, img_shape)).to(GPU.device) > 0.1
        # latents = perceiver(input_imgs, mask=mask)
        latents, _, _ = perceiver(input_imgs)

        x_hat = deceiver(latents, out_shape=img_shape)

        recon_loss = mse_loss(input_imgs, x_hat)
        recon_loss.backward()
        optimizer_recon.step()
        self.encoder_grads = helper.copy_grad_info(perceiver)

        self.logger.log_value_and_epoch_avg('train_recon_loss', recon_loss.item(), self.epoch, current_bs, self.data_len)

        # -------------------------------
        #     Regularization Training
        # --------------------------------
        optimizer_regul.zero_grad()

        # Sample noise as discriminator ground truth
        z = Tensor(np.random.normal(0, 1, (current_bs, perceiver.latent_dim))).to(GPU.device)

        # Measure discriminator's ability to recognize real samples
        predictions = discriminator(latents.detach())
        real_loss = adversarial_loss(predictions, valid)
        real_acc = Experiment.accuracy(predictions, valid, current_bs)

        # Measure discriminator's ability to recognize fake samples
        latents, _, _ = perceiver(deceiver(z, out_shape=img_shape))
        # torch.std(latents, dim=0)
        # latents = (latents - latents.mean(dim=0)) / torch.std(latents, dim=0)
        # predictions = discriminator(latents)
        predictions = discriminator(z)
        fake_loss = adversarial_loss(predictions, fake)
        fake_acc = Experiment.accuracy(predictions, fake, current_bs)

        regul_loss = 0.5 * (real_loss + fake_loss)
        acc = 0.5 * (real_acc + fake_acc)
        regul_loss.backward()
        optimizer_regul.step()

        self.logger.log_value_and_epoch_avg('train_loss_regularization', regul_loss.item(), self.epoch, current_bs, self.data_len)
        self.logger.log_value_and_epoch_avg('train_loss_real', real_loss.item(), self.epoch, current_bs, self.data_len)
        self.logger.log_value_and_epoch_avg('train_loss_fake', fake_loss.item(), self.epoch, current_bs, self.data_len)
        self.logger.log_value_and_epoch_avg('train_accuracy', acc, self.epoch, current_bs, self.data_len)
        self.logger.log_value_and_epoch_avg('train_real_accuracy', real_acc, self.epoch, current_bs, self.data_len)
        self.logger.log_value_and_epoch_avg('train_fake_accuracy', fake_acc, self.epoch, current_bs, self.data_len)

        return recon_loss + regul_loss, acc

    def __call__(self, setup: SetupStruct, epoch: int):
        models, losses, optimizers, schedulers = setup
        [model.train() for model in models]
        scheduler, scheduler_D = schedulers

        self.epoch = epoch
        self.grad_vis_done = not self.grad_vis

        self.logger.log_value('lr_encoder/decoder', scheduler.get_last_lr()[0], Training.global_step)
        self.logger.log_value('lr_D', scheduler_D.get_last_lr()[0], Training.global_step)

        running_loss = 0.0
        running_acc = 0.0

        valid = Tensor(self.dataloader.batch_size, 1).to(GPU.device).fill_(1.0).requires_grad_(False)
        fake = Tensor(self.dataloader.batch_size, 1).to(GPU.device).fill_(0.0).requires_grad_(False)

        pbar = tqdm(self.dataloader)
        pbar.set_description('Epoch {}'.format(epoch))

        sampling_ticker = self.data_len // self.sample_interval

        for i, imgs in enumerate(pbar):
            batches_done = epoch * self.data_len + i * imgs.shape[0]
            loss, accuracy = self.train(models, losses, optimizers, imgs, [valid, fake])

            if self.sample_interval is not None and batches_done % sampling_ticker == 0:
                self.sample_image(models, n_row=self.batch_size, batches_done=batches_done, epoch=epoch)
            self.logger.log_value_and_epoch_avg('train_loss', loss.item(), self.epoch, self.batch_size, self.data_len)
            self.logger.log_value_and_epoch_avg('train_accuracy', accuracy, self.epoch, self.batch_size, self.data_len)

            # log values
            self.logger.step(self.dataloader.batch_size)
            Training.global_step += self.dataloader.batch_size

            # calculate log values
            running_loss += loss.item() * self.dataloader.batch_size
            running_acc += accuracy * self.dataloader.batch_size


        # calculate log values
        epoch_loss = running_loss / len(self.data)
        epoch_acc = running_acc / len(self.data)

        self.logger.log_value('train_epoch_loss', epoch_loss, Training.global_step)
        self.logger.log_value('train_epoch_accuracy', epoch_acc, Training.global_step)

        # self.logger.log_value('train_epoch_accuracy', epoch_acc, Training.global_step)
        logging.info(f'Training {self.name} epoch {epoch} Lr: {scheduler.get_last_lr()[0]:.6f} Loss: {epoch_loss:.6f} D_Acc: {epoch_acc:.6f}')

        if not self.grad_vis_done:
            model_params_dict = {
                'Perceiver_Encoder': self.encoder_grads,
                'Deceiver_Decoder': models[1].named_parameters(),
                'Discriminator': models[2].named_parameters(),
            }
            helper.plot_multiple_grad_flow_lines(model_params_dict, f"{self.logger.log_dir}/data/Grad_flow_{self.epoch}.lines.png")
            self.grad_vis_done = True

        scheduler.step()
        scheduler_D.step()

        # do checkpointing
        if self.n_checkpoints is None or epoch % self.n_checkpoints == 0:
            helper.save_checkpoint(epoch, Training.global_step, models, optimizers, self.logger.log_dir, self.n_checkpoints)

        return  # do training

    def sample_image(self, models, n_row, batches_done, epoch):
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

            enc_img = perceiver(batch)
            batch = deceiver(enc_img, img_shape)
            batch = rearrange(batch, 'b h w c -> b c h w')

            batch_grid = make_grid(batch, nrow=n_row // 4)
            input_grid = make_grid(imgs, nrow=n_row // 4)

            batch_stack = torch.stack([input_grid, batch_grid])
            img_grid = make_grid(batch_stack, nrow=1)
            img_grid = helper.normalize(img_grid)
            save_image(img_grid.data, f"{self.logger.log_dir}/data/Reconstruction_{epoch}_{batches_done}.png", nrow=n_row)

            z = Tensor(np.random.normal(0, 1, (batch_size, perceiver.latent_dim))).to(GPU.device)
            gen_imgs = deceiver(z, out_shape=img_shape)
            gen_imgs = rearrange(gen_imgs, 'b h w c -> b c h w')

            save_image(gen_imgs.data, f"{self.logger.log_dir}/data/{self.name}_random_sample_train_{self.epoch}_{batches_done}.png", nrow=n_row // 8, normalize=True)
            plt.close()
        [model.train() for model in models]


class Validation:
    global_step = 0

    def __init__(self, data, batchsize, worker, logger, sample_interval, save_best_model=True, **kwargs):
        self.data = DataZoo.get(**data, **kwargs)
        self.batch_size = batchsize
        self.dataloader = DataLoader(self.data, batch_size=batchsize, shuffle=False, num_workers=worker)
        self.name = 'Impress_Perceiver_AT'
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

        input_imgs = rearrange(real_imgs, 'b c h w -> b h w c')
        img_shape = input_imgs.shape

        current_bs = self.dataloader.batch_size
        if real_imgs.shape[0] != self.dataloader.batch_size:
            current_bs = real_imgs.shape[0]
            valid = valid[0:current_bs, :]
            fake = fake[0:current_bs, :]

        # -------------------------
        #  Validate Reconstruction
        # -------------------------

        encoded_imgs = encoder(input_imgs)
        x_hat = decoder(encoded_imgs, out_shape=img_shape)
        self.logger.accumulate_embedding_set_for_epoch(encoded_imgs.detach().cpu(), labels, images=imgs, name=self.name)

        recon_loss = mse_loss(input_imgs, x_hat)
        self.logger.log_value_and_epoch_avg('val_loss_reconstruction', recon_loss.item(), self.epoch, current_bs, self.data_len)

        # -----------------------
        #  Validate Discriminator
        # -----------------------

        # Sample noise as discriminator ground truth
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], encoder.latent_dim))).to(GPU.device))

        # Measure discriminator's ability to recognize real images
        prediction_real = discriminator(encoded_imgs)
        real_loss = adv_loss(prediction_real, valid)
        accuracy_valid = Experiment.accuracy(prediction_real.detach(), valid, current_bs)
        # Measure discriminator's ability to recognize fake images
        prediction_fake = discriminator(z)
        fake_loss = adv_loss(prediction_fake, fake)
        accuracy_fake = Experiment.accuracy(prediction_fake.detach(), fake, current_bs)

        regul_loss = 0.5 * (real_loss + fake_loss)
        accuracy = 0.5 * (accuracy_fake + accuracy_valid)

        self.logger.log_value_and_epoch_avg('val_loss_regularization', regul_loss.item(), self.epoch, current_bs, self.data_len)
        self.logger.log_value_and_epoch_avg('val_loss_real', real_loss.item(), self.epoch, current_bs, self.data_len)
        self.logger.log_value_and_epoch_avg('val_loss_fake', fake_loss.item(), self.epoch, current_bs, self.data_len)

        self.logger.log_value_and_epoch_avg('val_accuracy', accuracy, self.epoch, current_bs, self.data_len)
        self.logger.log_value_and_epoch_avg('val_accuracy_real', accuracy_valid, self.epoch, current_bs, self.data_len)
        self.logger.log_value_and_epoch_avg('val_accuracy_fake', accuracy_fake, self.epoch, current_bs, self.data_len)

        return recon_loss + regul_loss, accuracy

    def __call__(self, setup, epoch):
        self.epoch = epoch
        models, losses, optimizers, _ = setup
        [model.eval() for model in models]

        pbar = tqdm(self.dataloader)
        pbar.set_description('Validation Epoch {}'.format(epoch))
        # Adversarial ground truths
        valid = Variable(Tensor(self.batch_size, 1).to(GPU.device).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(self.batch_size, 1).to(GPU.device).fill_(0.0), requires_grad=False)
        gt_vector = [valid, fake]

        running_loss = 0.0
        running_acc = 0.0
        sampling_ticker = self.data_len // self.sample_interval

        for i, data in enumerate(pbar):
            batches_done = epoch * self.data_len + i * data[0].shape[0]
            loss, accuracy = self.train(models, losses, data, gt_vector)

            if self.sample_interval is not None and batches_done % sampling_ticker == 0:
                self.sample_image(models, n_row=self.batch_size, batches_done=batches_done)

            # log values
            self.logger.step(self.batch_size)
            Validation.global_step += self.batch_size

            # calculate log values
            running_loss += loss.item() * self.dataloader.batch_size
            running_acc += accuracy * self.dataloader.batch_size
            self.logger.log_value_and_epoch_avg('validation_loss', loss.item(), self.epoch, self.batch_size, self.data_len)
            self.logger.log_value_and_epoch_avg('validation_accuracy', accuracy, self.epoch, self.batch_size, self.data_len)

        epoch_loss = running_loss / self.data_len
        epoch_acc = running_acc / self.data_len

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
            self.logger.log_value('Kmeans_accuracy', kmeans_acc)

            self.logger.log_embedding_set(self.name, step=epoch)
        else:
            self.logger.clear_embedding_set(self.name)

        logging.info(f'Validation {self.name} epoch {epoch} Loss: {epoch_loss:.6f} Accuracy: {epoch_acc:.6f} ClusterAccuracy: {kmeans_acc:.6f}')

        return

    def sample_image(self, models, n_row, batches_done):
        """Saves a grid of generated data"""
        with torch.no_grad():
            encoder, decoder, discriminator = models

            batch_start = ((batches_done * self.batch_size) % self.data_len)
            batch_end = ((batches_done * self.batch_size + self.batch_size) % self.data_len)
            imgs = self.data[batch_start:batch_end]

            # imgs = [item for sublist in zip([a for a, _ in imgs], [b for _, b in imgs]) for item in sublist]
            imgs = [img for _, img in imgs]
            imgs = torch.stack(imgs, dim=0)
            imgs = imgs.to(GPU.device)
            batch = rearrange(imgs, 'b c h w -> b h w c')
            img_shape = batch.shape

            z = Tensor(np.random.normal(0, 1, (self.batch_size, decoder.latent_dim))).to(GPU.device)
            gen_imgs = decoder(z, out_shape=img_shape)
            gen_imgs = rearrange(gen_imgs, 'b h w c -> b c h w')
            save_image(gen_imgs.data, f"{self.logger.log_dir}/data/random_sample_val_{self.epoch}_{batches_done}.png", nrow=n_row // 2 or n_row, normalize=True)

            batch = encoder(batch)
            skewed_encoding = batch.detach()
            for i in range(1, self.batch_size):
                skewed_encoding[i] += skewed_encoding[0]
            skewed_images = decoder(skewed_encoding, out_shape=img_shape)
            skewed_images = rearrange(skewed_images, 'b h w c -> b c h w')
            save_image(skewed_images.data, f"{self.logger.log_dir}/data/skewed_sample_val_{self.epoch}_{batches_done}.png", nrow=n_row // 2 or n_row, normalize=True)
            prediction = discriminator(batch.detach()).detach()
            prediction_bool = (prediction > 0.5).type(torch.cuda.FloatTensor)
            # prediction = prediction.repeat(1, pixels // 2).view(prediction_shape)
            prediction = einops.repeat(prediction, "b c -> b c h w", c=1, h=img_shape[1] // 2, w=img_shape[2])
            # prediction_bool = prediction_bool.repeat(1, pixels // 2).view(prediction_shape)
            prediction_bool = einops.repeat(prediction_bool, "b c -> b c h w", c=1, h=img_shape[1] // 2, w=img_shape[2])
            prediction_comb = torch.cat([prediction_bool, prediction], 2)
            prediction_grid = make_grid(prediction_comb, nrow=n_row // 8 or n_row)
            del prediction_comb, prediction_bool, prediction, skewed_images, gen_imgs, skewed_encoding

            batch = decoder(batch, out_shape=img_shape)
            batch = rearrange(batch, 'b h w c -> b c h w')
            input_grid = make_grid(imgs, nrow=n_row // 8 or n_row)
            batch_grid = make_grid(batch.detach(), nrow=n_row // 8 or n_row)
            batch_stack = torch.stack([input_grid, batch_grid, prediction_grid])
            # batch_stack = torch.stack([input_grid, batch_grid])

            img_grid = make_grid(batch_stack, nrow=1)
            img_grid = helper.normalize(img_grid)
            save_image(img_grid.data, f"{self.logger.log_dir}/data/Reconstruction_val_{self.epoch}_{batches_done}.png", nrow=n_row // 2 or n_row)
            # free up cuda memory
            del batch_stack, img_grid, batch_grid, input_grid, prediction_grid, batch, imgs, z, batch_start, batch_end