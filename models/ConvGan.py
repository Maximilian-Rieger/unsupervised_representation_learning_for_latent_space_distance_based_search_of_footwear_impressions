import torch
from torch import nn, Tensor
import numpy as np
from torch.autograd import Variable
from utils.utils import GPU
from models.ResUNet import ResDownBlock, ResUpBlock
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


class ConvEncoder(nn.Module):
    def __init__(self, img_shape, latent_size=100, activation=utils.Mish, activation_args=None):
        super(ConvEncoder, self).__init__()

        if activation_args is None:
            # activation_args = {'negative_slope': 0.2, 'inplace': True}
            activation_args = {}

        self.img_shape = img_shape
        self.latent_size = latent_size
        self.pixels = int(np.prod(self.img_shape[1:]) / 2)

        # encoder
        self.block_1 = ResDownBlock(img_shape[0], 32, activation=activation)
        self.block_2 = ResDownBlock(32, 64, activation=activation)
        self.block_3 = ResDownBlock(64, 128, activation=activation)
        self.block_4 = ResDownBlock(128, 256, activation=activation)
        self.block_5 = ResDownBlock(256, 512, activation=activation)

        self.fc_mu = nn.Linear(self.pixels, latent_size)
        self.fc_var = nn.Linear(self.pixels, latent_size)

        utils.init_weights(self.modules())

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = x.view(-1, self.pixels)
        mu, logvar = self.fc_mu(x), self.fc_var(x)
        z = sample(mu, logvar)
        return z


class ConvDecoder(nn.Module):
    def __init__(self, img_shape, latent_size=100, activation=utils.Mish, activation_args=None, last=nn.Tanh):
        super(ConvDecoder, self).__init__()
        if activation_args is None:
            activation_args = {}

        self.img_shape = img_shape
        self.latent_size = latent_size
        self.pixels = np.prod(self.img_shape)
        self.start_res = int(img_shape[-1] / 2 ** 6)
        # decoder
        self.fc_z = nn.Linear(latent_size, self.pixels)

        self.block_1 = ResUpBlock(int(self.pixels / self.start_res ** 2), 512)
        self.block_2 = ResUpBlock(512, 256, activation=activation)
        self.block_3 = ResUpBlock(256, 128, activation=activation)
        self.block_4 = ResUpBlock(128, 64, activation=activation)
        self.block_5 = ResUpBlock(64, 32, activation=activation)
        self.block_6 = ResUpBlock(32, img_shape[0], activation=activation)

        self.last = last()

        utils.init_weights(self.modules())

    def forward(self, z):
        z = self.fc_z(z)
        z = z.view(-1, int(self.pixels / self.start_res ** 2), self.start_res, self.start_res)
        z = self.block_1(z)
        z = self.block_2(z)
        z = self.block_3(z)
        z = self.block_4(z)
        z = self.block_5(z)
        z = self.block_6(z)
        return self.last(z)


class AutoGanDecoder(nn.Module):
    def __init__(self, img_shape, latent_size=100, activation=utils.Mish, activation_args=None, last=nn.Tanh):
        super(AutoGanDecoder, self).__init__()
        if activation_args is None:
            activation_args = {}

        self.img_shape = img_shape
        self.latent_size = latent_size
        self.pixels = np.prod(self.img_shape)

        # decoder
        self.fc_z = nn.Linear(latent_size, self.pixels)

        self.block_1 = ResUpBlock(self.pixels, 512)
        self.block_2 = ResUpBlock(512, 256, activation=activation)
        self.block_3 = ResUpBlock(256, 128, activation=activation)
        self.block_4 = ResUpBlock(128, 64, activation=activation)
        self.block_5 = ResUpBlock(64, 32, activation=activation)
        self.block_5 = ResUpBlock(32, img_shape[0], activation=activation)

        self.last = last()

        utils.init_weights(self.modules())

    def pre_activation_conv(self, in_size, out_size, stride, activation=utils.Mish, activation_args=None):
        block = []
        if activation_args is None:
            # activation_args = {'negative_slope': 0.2, 'inplace': True}
            activation_args = {}
        self.bn1 = nn.BatchNorm2d(in_size)
        self.act1 = activation(**activation_args)
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.act2 = activation(**activation_args)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)

    def forward(self, z):
        z = self.fc_z(z)
        z = z.view(-1, self.pixels, 32, 32)
        z = self.block_1(z)
        z = self.block_2(z)
        z = self.block_3(z)
        z = self.block_4(z)
        z = self.block_5(z)
        return self.last(z)


class ConvDiscriminator(nn.Module):
    def __init__(self, img_shape, last=nn.Sigmoid, activation=utils.Mish, activation_args=None):
        super(ConvDiscriminator, self).__init__()
        self.img_shape = img_shape
        self.pixels = int(np.prod(self.img_shape) / 2)

        self.block_1 = ResDownBlock(img_shape[0], 32, activation=activation)
        self.block_2 = ResDownBlock(32, 64, activation=activation)
        self.block_3 = ResDownBlock(64, 128, activation=activation)
        self.block_4 = ResDownBlock(128, 256, activation=activation)
        self.block_5 = ResDownBlock(256, 512, activation=activation)

        self.linear = nn.Linear(self.pixels, 1)
        self.last = last()

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = x.view(-1, self.pixels)
        x = self.linear(x)
        return x


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


class LinearBlock(nn.Module):
    def __init__(self, in_size, out_size, activation=utils.Mish, activation_args=None):
        super(LinearBlock, self).__init__()
        if activation_args is None:
            activation_args = {}
        self.linear = nn.Linear(in_size, out_size)
        self.act = activation(**activation_args)
        self.bn = nn.BatchNorm1d(out_size)

    def forward(self, z):
        z = self.linear(z)
        z = self.act(z)
        z = self.bn(z)
        return z


class Discriminator2(nn.Module):
    def __init__(self, latent_size=100, activation=utils.Mish, activation_args=None):
        super(Discriminator2, self).__init__()
        if activation_args is None:
            activation_args = {}

        self.latent_size = latent_size

        self.block_1 = LinearBlock(latent_size, 512)
        self.block_2 = LinearBlock(512, 256)
        self.block_3 = LinearBlock(256, 128)
        self.block_4 = LinearBlock(128, 64)
        self.block_5 = LinearBlock(64, 1)

        self.last = nn.Sigmoid()

        utils.init_weights(self.modules())

    def forward(self, z):
        z = self.block_1(z)
        z = self.block_2(z)
        z = self.block_3(z)
        z = self.block_4(z)
        z = self.block_5(z)

        return self.last(z)

