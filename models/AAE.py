import logging

import torch
from torch import nn, Tensor
import numpy as np
from torch.autograd import Variable
from utils.utils import GPU
from models.ResUNet import ResUNetDownBlock, ResUpBlock
import models.utils as utils

import torch.nn.functional as F


def reparameterization(mu, logvar, latent_size=100):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), latent_size))).to(GPU.device))
    z = sampled_z * std + mu
    return z


def sample(mu, logvar):
    std = torch.exp(0.5*logvar)  # e^(1/2 * log(std^2))
    eps = torch.randn_like(std)  # random ~ N(0, 1)
    return eps.mul(std).add_(mu)


class Encoder(nn.Module):
    def __init__(self, img_shape, latent_dim=10):
        super(Encoder, self).__init__()
        self.latent_size = latent_dim
        self.input_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mu = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar, self.latent_size)
        return z


class ConvEncoder(nn.Module):
    def __init__(self, img_shape, latent_size=10, activation=nn.LeakyReLU):
        super(ConvEncoder, self).__init__()
        self.latent_size = latent_size
        self.input_shape = img_shape

        self.conv_part = nn.Sequential(
            ResUNetDownBlock(img_shape[0], 2 ** 6, activation=activation),
            ResUNetDownBlock(2 ** 6, 2 ** 7, activation=activation),
        )

        img_size = int(img_shape[1] / 2 ** len(self.conv_part))

        self.model = nn.Sequential(
            nn.Linear(2 ** 7 * img_size ** 2, 2 ** 9),
            activation(0.2, inplace=True) if isinstance(activation, nn.LeakyReLU) else activation(),
            nn.Linear(2 ** 9, 2 ** 9),
            nn.BatchNorm1d(2 ** 9),
            activation(0.2, inplace=True) if isinstance(activation, nn.LeakyReLU) else activation(),
        )

        self.mu = nn.Linear(512, latent_size)
        self.logvar = nn.Linear(512, latent_size)

    def forward(self, img):
        # img_flat = img.view(img.shape[0], -1)
        x = self.conv_part(img)
        x = x.view(img.shape[0], -1)
        x = self.model(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar, self.latent_size)
        return z


class ConvEncoder2(nn.Module):
    def __init__(self, img_shape, latent_size=100, beta=1, activation=nn.LeakyReLU, activation_args=None):
        super(ConvEncoder2, self).__init__()
        self.img_shape = img_shape
        self.input_shape = img_shape

        if activation_args is None:
            # activation_args = {'negative_slope': 0.2, 'inplace': True}
            activation_args = {}

        self.latent_size = latent_size
        self.beta = beta

        self.pixels = int(np.prod(self.img_shape[1:]) / 2)

        # encoder
        self.conv_e1 = self._conv(img_shape[0], 32)
        self.act_e1 = activation(**activation_args)
        self.conv_e2 = self._conv(32, 64)
        self.act_e2 = activation(**activation_args)
        self.conv_e3 = self._conv(64, 128)
        self.act_e3 = activation(**activation_args)
        self.conv_e4 = self._conv(128, 256)
        self.act_e4 = activation(**activation_args)
        self.conv_e5 = self._conv(256, 512)
        self.act_e5 = activation(**activation_args)
        self.fc_mu = nn.Linear(self.pixels, latent_size)
        self.fc_var = nn.Linear(self.pixels, latent_size)

        utils.init_weights(self.modules())

    def _conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.conv_e1(x)
        x = self.act_e1(x)
        x = self.conv_e2(x)
        x = self.act_e2(x)
        x = self.conv_e3(x)
        x = self.act_e3(x)
        x = self.conv_e4(x)
        x = self.act_e4(x)
        x = self.conv_e5(x)
        x = self.act_e5(x)
        x = x.view(-1, self.pixels)
        mu, logvar = self.fc_mu(x), self.fc_var(x)
        z = sample(mu, logvar)
        return z


class ConvEncoder22(nn.Module):
    def __init__(self, img_shape, latent_size=100, beta=1, activation=nn.LeakyReLU, activation_args=None):
        super(ConvEncoder22, self).__init__()
        self.img_shape = img_shape
        self.input_shape = img_shape

        if activation_args is None:
            activation_args = {}

        self.latent_size = latent_size
        self.beta = beta

        self.pixels = int(np.prod(self.img_shape[1:]) / 2)

        # encoder
        self.conv_e1 = self._conv(img_shape[0], 32)
        self.act_e1 = activation(**activation_args)
        self.conv_e2 = self._conv(32, 64)
        self.act_e2 = activation(**activation_args)
        self.conv_e3 = self._conv(64, 128)
        self.act_e3 = activation(**activation_args)
        self.conv_e4 = self._conv(128, 256)
        self.act_e4 = activation(**activation_args)
        self.conv_e5 = self._conv(256, 512)
        self.act_e5 = activation(**activation_args)
        self.fc_mu = nn.Linear(self.pixels, latent_size)
        self.fc_var = nn.Linear(self.pixels, latent_size)
        self.act_eLast = activation(**activation_args)

        utils.init_weights(self.modules())

    def _conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.conv_e1(x)
        x = self.act_e1(x)
        x = self.conv_e2(x)
        x = self.act_e2(x)
        x = self.conv_e3(x)
        x = self.act_e3(x)
        x = self.conv_e4(x)
        x = self.act_e4(x)
        x = self.conv_e5(x)
        x = self.act_e5(x)
        x = x.view(-1, self.pixels)
        mu, logvar = self.fc_mu(x), self.fc_var(x)
        mu, logvar = self.act_eLast(mu), self.act_eLast(logvar)
        z = sample(mu, logvar)
        return z


class ConvEncoder3(nn.Module):
    def __init__(self, img_shape, latent_size=100, beta=1, activation=nn.LeakyReLU, activation_args=None):
        super(ConvEncoder3, self).__init__()
        self.img_shape = img_shape
        self.input_shape = img_shape

        if activation_args is None:
            # activation_args = {'negative_slope': 0.2, 'inplace': True}
            activation_args = {}

        self.latent_size = latent_size
        self.beta = beta

        self.pixels = int(np.prod(self.img_shape[1:]) / 2)

        # encoder normal path
        self.conv_e1 = self._conv(img_shape[0], 32)
        self.act_e1 = activation(**activation_args)
        self.conv_e2 = self._conv(32, 64)
        self.act_e2 = activation(**activation_args)
        self.conv_e3 = self._conv(64, 128)
        self.act_e3 = activation(**activation_args)

        self.conv_e1_skip = self._conv(img_shape[0], 128, 8)
        self.act_e1_skip = activation(**activation_args)

        self.conv_e4 = self._conv(128, 256)
        self.act_e4 = activation(**activation_args)
        self.conv_e5 = self._conv(256, 512)
        self.act_e5 = activation(**activation_args)

        self.conv_e2_skip = self._conv(128, 512, 4)
        self.act_e2_skip = activation(**activation_args)

        self.fc_mu = nn.Linear(self.pixels, latent_size)
        self.fc_var = nn.Linear(self.pixels, latent_size)

        utils.init_weights(self.modules())

    def _conv(self, in_channels, out_channels, stride=2):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=4, stride=stride, padding=1
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        y = self.conv_e1_skip(x)
        y = self.act_e1_skip(y)

        x = self.conv_e1(x)
        x = self.act_e1(x)

        x = self.conv_e2(x)
        x = self.act_e2(x)

        x = self.conv_e3(x)
        x = self.act_e3(x)

        x = x + y

        y = self.conv_e2_skip(y)
        y = self.act_e2_skip(y)
        x = self.conv_e4(x)
        x = self.act_e4(x)
        x = self.conv_e5(x)
        x = self.act_e5(x)
        x = x + y
        x = x.view(-1, self.pixels)
        mu, logvar = self.fc_mu(x), self.fc_var(x)
        z = sample(mu, logvar)
        return z


class Decoder(nn.Module):
    def __init__(self, img_shape, latent_size=10):
        super(Decoder, self).__init__()
        self.img_shape = img_shape
        self.latent_size = latent_size
        self.input_shape = latent_size
        self.model = nn.Sequential(
            nn.Linear(latent_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *self.img_shape)
        return img


class ConvDecoder(nn.Module):
    def __init__(self, img_shape, latent_size=10, activation=nn.LeakyReLU):
        super(ConvDecoder, self).__init__()
        self.img_shape = img_shape
        self.input_shape = img_shape
        self.latent_size = latent_size

        self.conv_part = nn.Sequential(
            ResUpBlock(256, 128, activation=activation),
            ResUpBlock(128, img_shape[0], activation=activation),
        )

        self.img_size = int(img_shape[1] / 2 ** len(self.conv_part))

        self.model = nn.Sequential(
            nn.Linear(latent_size, 512),
            activation(0.2, inplace=True) if isinstance(activation, nn.LeakyReLU) else activation(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            activation(0.2, inplace=True) if isinstance(activation, nn.LeakyReLU) else activation(),
            nn.Linear(512, int(256 * self.img_size ** 2)),
        )

        self.last = nn.Tanh()

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *(self.img_shape[0], self.img_size, self.img_size))
        img = self.conv_part(img)
        img = self.last(img)
        return img


class ConvDecoder2(nn.Module):
    def __init__(self, img_shape, latent_size=100, beta=1, activation=nn.LeakyReLU, activation_args=None, last=nn.Tanh):
        super(ConvDecoder2, self).__init__()
        self.img_shape = img_shape
        self.input_shape = img_shape

        if activation_args is None:
            # activation_args = {'negative_slope': 0.2, 'inplace': True}
            activation_args = {}

        self.latent_size = latent_size
        self.beta = beta

        self.pixels = np.prod(self.img_shape)

        # decoder
        self.fc_z = nn.Linear(latent_size, self.pixels)
        self.conv_d1 = self._upconv(2 * 2048 * self.img_shape[0], 512)
        self.act_d1 = activation(**activation_args)
        self.conv_d2 = self._upconv(512, 256)
        self.act_d2 = activation(**activation_args)
        self.conv_d3 = self._upconv(256, 128)
        self.act_d3 = activation(**activation_args)
        self.conv_d4 = self._upconv(128, 64)
        self.act_d4 = activation(**activation_args)
        self.conv_d5 = self._upconv(64, 32)
        self.act_d5 = activation(**activation_args)
        self.conv_d6 = self._upconv(32, img_shape[0])
        # self.act_d6 = activation(**activation_args) # WTF !?

        self.last = last()

        utils.init_weights(self.modules())

    def _upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, stride=1
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, z):
        z = self.fc_z(z)
        z = z.view(-1, 2 * 2048 * self.img_shape[0], 4, 4)
        z = F.interpolate(z, scale_factor=2)
        z = self.conv_d1(z)
        z = self.act_d1(z)
        z = F.interpolate(z, scale_factor=2)
        z = self.conv_d2(z)
        z = self.act_d2(z)
        z = F.interpolate(z, scale_factor=2)
        z = self.conv_d3(z)
        z = self.act_d3(z)
        z = F.interpolate(z, scale_factor=2)
        z = self.conv_d4(z)
        z = self.act_d4(z)
        z = F.interpolate(z, scale_factor=2)
        z = self.conv_d5(z)
        z = self.act_d5(z)
        z = F.interpolate(z, scale_factor=2)
        z = self.conv_d6(z)
        # z = self.act_d6(z) # wtf !?
        return self.last(z)


class ConvDecoder3(nn.Module):
    def __init__(self, img_shape, latent_size=100, beta=1, activation=nn.LeakyReLU, activation_args=None, last=nn.Tanh):
        super(ConvDecoder3, self).__init__()
        self.img_shape = img_shape
        self.input_shape = img_shape

        if activation_args is None:
            activation_args = {}

        self.latent_size = latent_size
        self.beta = beta

        self.pixels = np.prod(self.img_shape)

        # decoder normal path
        self.fc_z = nn.Linear(latent_size, self.pixels)
        self.conv_d1 = self._upconv(2 * 2048 * self.img_shape[0], 512)
        self.act_d1 = activation(**activation_args)
        self.conv_d2 = self._upconv(512, 256)
        self.act_d2 = activation(**activation_args)

        self.conv_d1_skip = self._upconv(2 * 2048 * self.img_shape[0], 256)
        self.act_d1_skip = activation(**activation_args)

        self.conv_d3 = self._upconv(256, 128)
        self.act_d3 = activation(**activation_args)
        self.conv_d4 = self._upconv(128, 64)
        self.act_d4 = activation(**activation_args)
        self.conv_d5 = self._upconv(64, 32)

        self.conv_d2_skip = self._upconv(256, 32)
        self.act_d2_skip = activation(**activation_args)

        self.act_d5 = activation(**activation_args)
        self.conv_d6 = self._upconv(32, img_shape[0])

        self.last = last()

        utils.init_weights(self.modules())

    def _upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, stride=1
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, z):
        z = self.fc_z(z)
        z = z.view(-1, 2 * 2048 * self.img_shape[0], 4, 4)
        z = F.interpolate(z, scale_factor=2)
        y = z.clone()
        y = self.conv_d1_skip(y)
        y = self.act_d1_skip(y)
        y = F.interpolate(y, scale_factor=4)
        z = self.conv_d1(z)
        z = self.act_d1(z)
        z = F.interpolate(z, scale_factor=2)
        z = self.conv_d2(z)
        z = self.act_d2(z)
        z = F.interpolate(z, scale_factor=2)
        z = z + y
        y = self.conv_d2_skip(y)
        y = self.act_d2_skip(y)
        y = F.interpolate(y, scale_factor=8)
        z = self.conv_d3(z)
        z = self.act_d3(z)
        z = F.interpolate(z, scale_factor=2)
        z = self.conv_d4(z)
        z = self.act_d4(z)
        z = F.interpolate(z, scale_factor=2)
        z = self.conv_d5(z)
        z = self.act_d5(z)
        z = F.interpolate(z, scale_factor=2)
        z = z + y
        z = self.conv_d6(z)
        return self.last(z)


class Discriminator(nn.Module):
    def __init__(self, latent_size=100, activation=nn.ReLU, activation_args=None):
        super(Discriminator, self).__init__()
        self.latent_size = latent_size
        self.input_shape = latent_size
        if activation_args is None:
            activation_args = {}

        self.model = nn.Sequential(
            nn.Linear(latent_size, 512),
            activation(**activation_args),
            nn.Linear(512, 256),
            activation(**activation_args),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity


class Discriminator2(nn.Module):
    def __init__(self, latent_size=100, beta=1, activation=nn.ReLU, activation_args=None):
        super(Discriminator2, self).__init__()
        if activation_args is None:
            activation_args = {}

        self.latent_size = latent_size
        self.input_shape = latent_size
        self.beta = beta

        self.model = nn.Sequential(
            nn.Linear(latent_size, 512),
            activation(**activation_args),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            activation(**activation_args),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            activation(**activation_args),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            activation(**activation_args),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        utils.init_weights(self.modules())

    def forward(self, z):
        validity = self.model(z)
        return validity


class Discriminator3(nn.Module):
    def __init__(self, latent_size=100, beta=1, activation=nn.ReLU, activation_args=None, skip=False):
        super(Discriminator3, self).__init__()
        if activation_args is None:
            activation_args = {}

        self.latent_size = latent_size
        self.input_shape = latent_size
        self.beta = beta
        self.skip = skip

        self.linear1 = nn.Linear(latent_size, 256)
        self.act1 = activation(**activation_args)
        self.bn1 = nn.BatchNorm1d(256)
        self.linear2 = nn.Linear(256, 128)
        self.act2 = activation(**activation_args)
        self.bn2 = nn.BatchNorm1d(128)
        self.linear3 = nn.Linear(128, 1)
        self.last = nn.Sigmoid()

        if self.skip:
            self.skip_linear1 = nn.Linear(latent_size, 128)
            self.skip_act1 = activation(**activation_args)
            self.skip_bn1 = nn.BatchNorm1d(128)

        utils.init_weights(self.modules())

    def forward(self, z):
        if self.skip:
            y = self.skip_linear1(z)
            y = self.skip_act1(y)
            y = self.skip_bn1(y)
        z = self.linear1(z)
        z = self.act1(z)
        z = self.bn1(z)
        z = self.linear2(z)
        z = self.act2(z)
        z = self.bn2(z)
        if self.skip:
            z = z + y
        z = self.linear3(z)
        z = self.last(z)
        return z


class Discriminator4(nn.Module):
    def __init__(self, latent_size=100, beta=1, activation=nn.ReLU, activation_args=None, dropout=0.2):
        super(Discriminator4, self).__init__()
        if activation_args is None:
            activation_args = {}

        self.latent_size = latent_size
        self.input_shape = latent_size
        self.beta = beta
        self.dropout = dropout

        self.linear1 = nn.Linear(latent_size, 256)
        self.act1 = activation(**activation_args)
        self.bn1 = nn.BatchNorm1d(256)
        self.linear2 = nn.Linear(256, 128)
        self.act2 = activation(**activation_args)
        self.bn2 = nn.BatchNorm1d(128)
        self.linear3 = nn.Linear(128, 1)
        self.last = nn.Sigmoid()

        if self.dropout is not None:
            self.dp1 = nn.Dropout(self.dropout)
            self.dp2 = nn.Dropout(self.dropout)

        utils.init_weights(self.modules())

    def forward(self, z):
        z = self.linear1(z)
        z = self.act1(z)
        if self.dropout is None:
            z = self.bn1(z)
        else:
            z = self.dp1(z)
        z = self.linear2(z)
        z = self.act2(z)
        if self.dropout is None:
            z = self.bn2(z)
        else:
            z = self.dp2(z)
        z = self.linear3(z)
        z = self.last(z)
        return z

