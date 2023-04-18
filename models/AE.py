import torch
from torch import nn, Tensor
import numpy as np
from torch.autograd import Variable
from utils.utils import GPU
from models.ResUNet import ResUNetDownBlock, ResUpBlock
import models.utils as utils

import torch.nn.functional as F


class ConvEncoder(nn.Module):
    def __init__(self, img_shape, latent_size=10, activation=nn.LeakyReLU):
        super(ConvEncoder, self).__init__()
        self.latent_size = latent_size

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

        self.last = nn.Linear(512, latent_size)

    def forward(self, img):
        x = self.conv_part(img)
        x = x.view(img.shape[0], -1)
        x = self.model(x)
        x = self.last(x)
        return x


class ConvEncoder2_legacy(nn.Module):
    def __init__(self, img_shape, latent_size=100, beta=1, activation=nn.LeakyReLU, last=nn.LeakyReLU, activation_args=None):
        super(ConvEncoder2_legacy, self).__init__()
        self.img_shape = img_shape

        if activation_args is None:
            # activation_args = {'negative_slope': 0.2, 'inplace': True}
            activation_args = {}

        self.latent_size = latent_size
        self.beta = beta

        self.pixels = int(np.prod(self.img_shape[1:]) / 8)

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
        self.conv_e6 = self._conv(512, 512)
        self.act_e6 = activation(**activation_args)
        self.last = nn.Linear(self.pixels, latent_size)
        self.act_last = last(**activation_args)

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
        x = self.conv_e6(x)
        x = self.act_e6(x)
        x = x.view(-1, self.pixels)
        x = self.last(x)
        x = self.act_last(x)
        return x


class ConvEncoder2(nn.Module):
    def __init__(self, img_shape, latent_size=100, beta=1, activation=nn.LeakyReLU, last=nn.LeakyReLU, activation_args=None):
        super(ConvEncoder2, self).__init__()
        self.img_shape = img_shape

        if activation_args is None:
            # activation_args = {'negative_slope': 0.2, 'inplace': True}
            activation_args = {}

        self.latent_size = latent_size
        self.beta = beta

        self.pixels = int(np.prod(self.img_shape[1:]) / 8)

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
        self.conv_e6 = self._conv(512, 512)
        self.act_e6 = activation(**activation_args)
        self.fc_z = nn.Linear(self.pixels, latent_size)
        self.act_last = last(**activation_args)

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
        x = self.conv_e6(x)
        x = self.act_e6(x)
        x = x.view(-1, self.pixels)
        x = self.fc_z(x)
        x = self.act_last(x)
        return x


class Decoder(nn.Module):
    def __init__(self, img_shape, latent_size=10):
        super(Decoder, self).__init__()
        self.img_shape = img_shape
        self.latent_size = latent_size
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
    def __init__(self, img_shape, latent_size=100, beta=1, activation=nn.LeakyReLU, activation_args=None):
        super(ConvDecoder2, self).__init__()
        self.img_shape = img_shape

        if activation_args is None:
            # activation_args = {'negative_slope': 0.2, 'inplace': True}
            activation_args = {}

        self.latent_size = latent_size
        self.beta = beta

        self.pixels = int(np.prod(self.img_shape[1:]) / 4)

        # decoder
        self.fc_z = nn.Linear(latent_size, self.pixels)
        self.conv_d1 = self._upconv(1024 * self.img_shape[0], 512)
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
        self.act_d6 = activation(**activation_args)

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
        z = z.view(-1, 1024 * self.img_shape[0], 2, 2)
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
        z = self.act_d6(z)
        return z


class Discriminator(nn.Module):
    def __init__(self, latent_size=10):
        super(Discriminator, self).__init__()
        self.latent_size = latent_size

        self.model = nn.Sequential(
            nn.Linear(latent_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity


class Discriminator2(nn.Module):
    def __init__(self, latent_size=100, beta=1, activation=nn.LeakyReLU, activation_args=None):
        super(Discriminator2, self).__init__()
        if activation_args is None:
            # activation_args = {'negative_slope': 0.2, 'inplace': True}
            activation_args = {}

        self.latent_size = latent_size
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
    def __init__(self, latent_size=100, beta=1, activation=nn.LeakyReLU, activation_args=None):
        super(Discriminator3, self).__init__()
        if activation_args is None:
            # activation_args = {'negative_slope': 0.2, 'inplace': True}
            activation_args = {}

        self.latent_size = latent_size
        self.beta = beta

        self.model = nn.Sequential(
            nn.Linear(latent_size, 512),
            activation(**activation_args),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            activation(**activation_args),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

        utils.init_weights(self.modules())

    def forward(self, z):
        validity = self.model(z)
        return validity

