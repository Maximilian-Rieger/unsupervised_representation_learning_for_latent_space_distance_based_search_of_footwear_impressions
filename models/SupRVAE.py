import torch
import logging
from torch import nn
import numpy as np
import torch.nn.functional as F
from models.ResUNet import ResDownBlock, ResUpBlock
from models.ResUNet import ShortConvBlock, ConvBlock, LongConvBlock
import models.utils as utils


# AE

class ConvDirEncoder(nn.Module):
    def __init__(self, img_shape, latent_size=100, depth=3, start_filter=16, activation=utils.Mish, last=utils.Mish, activation_args=None, residual=True):
        super(ConvDirEncoder, self).__init__()
        logging.debug('ConvDirEncoder parameters: {}\n'.format({
            'img_shape': img_shape,
            'latent_size': latent_size,
            'depth': depth,
            'start_filter': start_filter,
        }))
        if activation_args is None:
            activation_args = {}
        self.img_shape = img_shape
        self.input_shape = img_shape
        self.latent_size = latent_size
        self.pixels = int(np.prod(self.img_shape))

        self.depth = depth

        current_units = start_filter
        # encoder

        self.first = nn.Conv2d(img_shape[0], current_units, 3, stride=1)

        self.blocks = []
        self.blocks.append(ResDownBlock(current_units, current_units, activation=activation, activation_args=activation_args, shortcut=residual))
        logging.debug('Encoder level {} | In {} Out {}'.format(0, img_shape[0], current_units))

        for i in range(depth - 1):
            previous_units = current_units
            current_units = current_units * 2
            self.blocks.append(ResDownBlock(previous_units, current_units, activation=activation, activation_args=activation_args, shortcut=residual))
            logging.debug('Encoder level {} | In {} Out {}'.format(i + 1, previous_units, current_units))

        self.blocks = nn.ModuleList(self.blocks)
        logging.debug('Encoder Modules {} '.format(list(self.modules())))

        self.middle = activation(**activation_args)

        self.fc = nn.Linear(self.pixels, latent_size)
        self.last = last()
        utils.init_weights(self.modules())

    def forward(self, x: torch.Tensor):
        x = self.first(x)
        # logging.debug('\nSupRVAE Encoder Input shape {}'.format(x.shape))
        for i in range(self.depth):
            # logging.debug('SupRVAE Encoder level {} | Input shape {}'.format(i, x.shape))
            x = self.blocks[i](x)
        # logging.debug('SupRVAE Encoder level {} | Input shape {}'.format(self.depth, x.shape))
        x = x.view(-1, self.pixels)
        x = self.middle(x)
        z = self.fc(x)
        z = self.last(z)
        # logging.debug('SupRVAE Encoder final output shape {}'.format(z.shape))
        return z


class ConvDirDecoder(nn.Module):
    def __init__(self, img_shape, latent_size=100, depth=3, end_filter=8, start_pixels=32, activation=utils.Mish, activation_args=None, last=nn.Tanh, residual=True):
        super(ConvDirDecoder, self).__init__()
        logging.debug('ConvDirDecoder parameters: {}\n'.format({
            'img_shape': img_shape,
            'latent_size': latent_size,
            'depth': depth,
            'end_filter': end_filter,
            'start_pixels': start_pixels,
        }))
        if activation_args is None:
            activation_args = {}

        self.img_shape = img_shape
        self.latent_size = latent_size
        self.input_shape = latent_size
        self.pixels = int(np.prod(self.img_shape))

        # decoder
        self.fc_z = nn.Linear(latent_size, self.pixels)

        self.depth = depth
        self.min_pixels = start_pixels
        self.start_pixels = start_pixels ** 2

        self.depth = depth
        current_units = end_filter

        self.blocks = []
        self.blocks.append(ResUpBlock(current_units, img_shape[0], activation=activation, activation_args=activation_args, shortcut=residual))
        logging.debug('Decoder level {} | In {} Out {}'.format(depth, current_units, img_shape[0]))

        for i in reversed(range(depth - 2)):
            previous_units = current_units
            current_units = current_units * 2
            self.blocks.append(ResUpBlock(current_units, previous_units, activation=activation, activation_args=activation_args, shortcut=residual))
            logging.debug('Decoder level {} | In {} Out {}'.format(i + 1, current_units, previous_units))

        self.max_filter = current_units * 2
        self.start_res = int(self.pixels / self.start_pixels / self.max_filter)
        self.blocks.append(ResUpBlock(self.start_res * self.max_filter, current_units, activation=activation, activation_args=activation_args, shortcut=residual))
        logging.debug('Decoder level {} | In {} Out {}'.format(0, self.start_res * self.max_filter, current_units))

        self.blocks = nn.ModuleList(reversed(self.blocks))
        logging.debug('Decoder Modules {} '.format(list(self.modules())))

        self.last = last()

        utils.init_weights(self.modules())

    def forward(self, z: torch.Tensor):
        # logging.debug('\nSupRVAE Decoder Input shape {}'.format(z.shape))
        z = self.fc_z(z)
        z = z.view(-1, self.start_res * self.max_filter, self.min_pixels, self.min_pixels)
        for i in range(self.depth):
            # logging.debug('SupRVAE Decoder level {} | Input shape {}'.format(i, z.shape))
            z = self.blocks[i](z)
        # logging.debug('SupRVAE Decoder level {}  | Input shape {}'.format(self.depth, z.shape))

        z = self.last(z)
        # logging.debug('SupRVAE Decoder final output shape {}'.format(z.shape))
        return z


# VAE


class ConvEncoder_simplified(nn.Module):
    def __init__(self,
                 img_shape: tuple,
                 latent_size: int = 100,
                 depth: int = 3,
                 activation: nn.Module = nn.ReLU,
                 activation_args: dict = None,
                 residual: bool = True,
                ):
        super(ConvEncoder_simplified, self).__init__()
        logging.debug('ConvEncoder parameters: {}\n'.format({
            'img_shape': img_shape,
            'latent_size': latent_size,
            'depth': depth,
        }))
        if activation_args is None:
            activation_args = {}
        self.img_shape = img_shape
        self.latent_size = latent_size
        self.pixels = int(np.prod(self.img_shape))

        self.depth = depth

        current_units = 4
        # encoder

        self.first = nn.Conv2d(img_shape[0], current_units, 3, stride=1)
        self.middle = activation(**activation_args)
        self.last = activation(**activation_args)
        self.blocks = []
        logging.debug('Encoder level {} | In {} Out {}'.format(0, img_shape[0], current_units))

        for i in range(depth):
            previous_units = current_units
            current_units = current_units * 2
            self.blocks.append(ResDownBlock(previous_units, current_units, activation=activation, activation_args=activation_args, shortcut=residual))
            logging.debug('Encoder level {} | In {} Out {}'.format(i + 1, previous_units, current_units))

        self.blocks = nn.ModuleList(self.blocks)
        logging.debug('Encoder Modules {} '.format(list(self.modules())))

        self.fc_mu = nn.Linear(self.pixels, latent_size)
        self.fc_var = nn.Linear(self.pixels, latent_size)

        utils.init_weights(self.modules())

    def forward(self, x: torch.Tensor):
        x = self.first(x)
        # logging.debug('\nSupRVAE Encoder Input shape {}'.format(x.shape))
        for i in range(self.depth):
            # logging.debug('SupRVAE Encoder level {} | Input shape {}'.format(i, x.shape))
            x = self.blocks[i](x)
        # logging.debug('SupRVAE Encoder level {} | Input shape {}'.format(self.depth, x.shape))
        x = x.view(-1, self.pixels)
        x = self.middle(x)
        mu, logvar = self.fc_mu(x), self.fc_var(x)
        x = self.last(x)
        z = utils.sample(mu, logvar)
        # logging.debug('SupRVAE Encoder final output shape {}'.format(z.shape))
        return z


class ConvEncoder(nn.Module):
    def __init__(self, img_shape, latent_size=100, depth=3, start_filter=16, activation=utils.Mish, activation_args=None, last=None, residual=True):
        super(ConvEncoder, self).__init__()
        logging.debug('ConvEncoder parameters: {}\n'.format({
            'img_shape': img_shape,
            'latent_size': latent_size,
            'depth': depth,
            'start_filter': start_filter,
        }))
        if activation_args is None:
            activation_args = {}
        if last is None:
            last = activation
        self.img_shape = img_shape
        self.input_shape = img_shape
        self.latent_size = latent_size
        self.scale_factor = 32 / start_filter
        self.pixels = int(np.prod(self.img_shape) / self.scale_factor)

        self.depth = depth

        current_units = start_filter
        # encoder

        self.first = nn.Conv2d(img_shape[0], current_units, 3, stride=1)

        self.blocks = []
        self.blocks.append(ResDownBlock(current_units, current_units, activation=activation, activation_args=activation_args, shortcut=residual))
        logging.debug('Encoder level {} | In {} Out {}'.format(0, img_shape[0], current_units))

        for i in range(depth - 1):
            previous_units = current_units
            current_units = current_units * 2
            self.blocks.append(ResDownBlock(previous_units, current_units, activation=activation, activation_args=activation_args, shortcut=residual))
            logging.debug('Encoder level {} | In {} Out {}'.format(i + 1, previous_units, current_units))

        self.blocks = nn.ModuleList(self.blocks)
        logging.debug('Encoder Modules {} '.format(list(self.modules())))

        self.middle = activation(**activation_args)

        # self.fc_mu = nn.Sequential(activation(), nn.Linear(self.pixels, latent_size))
        # self.fc_var = nn.Sequential(activation(), nn.Linear(self.pixels, latent_size))
        self.fc_mu = nn.Linear(self.pixels, latent_size)
        self.fc_var = nn.Linear(self.pixels, latent_size)
        self.last = last()
        utils.init_weights(self.modules())

    def forward(self, x: torch.Tensor):
        x = self.first(x)
        # logging.debug('\nSupRVAE Encoder Input shape {}'.format(x.shape))
        for i in range(self.depth):
            # logging.debug('SupRVAE Encoder level {} | Input shape {}'.format(i, x.shape))
            x = self.blocks[i](x)
        # logging.debug('SupRVAE Encoder level {} | Input shape {}'.format(self.depth, x.shape))
        x = x.view(-1, self.pixels)
        x = self.middle(x)
        mu, logvar = self.fc_mu(x), self.fc_var(x)
        mu, logvar = self.last(mu), self.last(logvar)
        z = utils.sample(mu, logvar)
        # logging.debug('SupRVAE Encoder final output shape {}'.format(z.shape))
        return z


class ConvDecoder(nn.Module):
    def __init__(self, img_shape, latent_size=100, depth=3, end_filter=8, start_pixels=32, activation=utils.Mish, activation_args=None, last=nn.Tanh, residual=True):
        super(ConvDecoder, self).__init__()
        logging.debug('ConvDecoder parameters: {}\n'.format({
            'img_shape': img_shape,
            'latent_size': latent_size,
            'depth': depth,
            'end_filter': end_filter,
            'start_pixels': start_pixels,
        }))
        if activation_args is None:
            activation_args = {}

        self.img_shape = img_shape
        self.latent_size = latent_size
        self.input_shape = latent_size
        self.pixels = int(np.prod(self.img_shape))

        # decoder
        self.fc_z = nn.Linear(latent_size, self.pixels)

        self.depth = depth
        self.min_pixels = start_pixels
        self.start_pixels = start_pixels ** 2

        self.depth = depth
        current_units = end_filter

        self.blocks = []
        self.blocks.append(ResUpBlock(current_units, img_shape[0], activation=activation, activation_args=activation_args, shortcut=residual))
        logging.debug('Decoder level {} | In {} Out {}'.format(depth, current_units, img_shape[0]))

        for i in reversed(range(depth - 2)):
            previous_units = current_units
            current_units = current_units * 2
            self.blocks.append(ResUpBlock(current_units, previous_units, activation=activation, activation_args=activation_args, shortcut=residual))
            logging.debug('Decoder level {} | In {} Out {}'.format(i + 1, current_units, previous_units))

        self.max_filter = current_units * 2
        self.start_res = int(self.pixels / self.start_pixels / self.max_filter)
        self.blocks.append(ResUpBlock(self.start_res * self.max_filter, current_units, activation=activation, activation_args=activation_args, shortcut=residual))
        logging.debug('Decoder level {} | In {} Out {}'.format(0, self.start_res * self.max_filter, current_units))

        self.blocks = nn.ModuleList(reversed(self.blocks))
        logging.debug('Decoder Modules {} '.format(list(self.modules())))

        self.last = last()

        utils.init_weights(self.modules())

    def forward(self, z: torch.Tensor):
        # logging.debug('\nSupRVAE Decoder Input shape {}'.format(z.shape))
        z = self.fc_z(z)
        z = z.view(-1, self.start_res * self.max_filter, self.min_pixels, self.min_pixels)
        for i in range(self.depth):
            # logging.debug('SupRVAE Decoder level {} | Input shape {}'.format(i, z.shape))
            z = self.blocks[i](z)
        # logging.debug('SupRVAE Decoder level {}  | Input shape {}'.format(self.depth, z.shape))

        z = self.last(z)
        # logging.debug('SupRVAE Decoder final output shape {}'.format(z.shape))
        return z


class ConvParEncoder(nn.Module):
    conv_block_mapping = {
        1: ShortConvBlock,
        2: ConvBlock,
        3: LongConvBlock,
    }

    def __init__(self, img_shape, latent_size=100, depth=3, start_filter=16, activation=utils.Mish, activation_args=None, last=None, residual=True, conv_block_length=2):
        super(ConvParEncoder, self).__init__()
        logging.debug('ConvEncoder parameters: {}\n'.format({
            'img_shape': img_shape,
            'latent_size': latent_size,
            'depth': depth,
            'start_filter': start_filter,
        }))
        if activation_args is None:
            activation_args = {}
        if last is None:
            last = activation
        assert conv_block_length in ConvParEncoder.conv_block_mapping.keys(), 'Convblock length must be a known length'
        conv_block = ConvParEncoder.conv_block_mapping[conv_block_length]

        self.img_shape = img_shape
        self.input_shape = img_shape
        self.latent_size = latent_size
        self.pixels = int(np.prod(self.img_shape))

        self.depth = depth

        current_units = start_filter
        # encoder

        self.first = nn.Conv2d(img_shape[0], current_units, 3, stride=1)

        self.blocks = []
        self.blocks.append(ResDownBlock(current_units, current_units, activation=activation, activation_args=activation_args, shortcut=residual, conv_block=conv_block))
        logging.debug('Encoder level {} | In {} Out {}'.format(0, img_shape[0], current_units))

        for i in range(depth - 1):
            previous_units = current_units
            current_units = current_units * 2
            self.blocks.append(ResDownBlock(previous_units, current_units, activation=activation, activation_args=activation_args, shortcut=residual, conv_block=conv_block))
            logging.debug('Encoder level {} | In {} Out {}'.format(i + 1, previous_units, current_units))

        self.blocks = nn.ModuleList(self.blocks)
        logging.debug('Encoder Modules {} '.format(list(self.modules())))

        self.middle = activation(**activation_args)

        # self.fc_mu = nn.Sequential(activation(), nn.Linear(self.pixels, latent_size))
        # self.fc_var = nn.Sequential(activation(), nn.Linear(self.pixels, latent_size))
        self.fc_mu = nn.Linear(self.pixels, latent_size)
        self.fc_var = nn.Linear(self.pixels, latent_size)
        self.last = last()
        utils.init_weights(self.modules())
        self.mu, self.logvar = None, None

    def forward(self, x: torch.Tensor):
        x = self.first(x)
        # logging.debug('\nSupRVAE Encoder Input shape {}'.format(x.shape))
        for i in range(self.depth):
            # logging.debug('SupRVAE Encoder level {} | Input shape {}'.format(i, x.shape))
            x = self.blocks[i](x)
        # logging.debug('SupRVAE Encoder level {} | Input shape {}'.format(self.depth, x.shape))
        x = x.view(-1, self.pixels)
        x = self.middle(x)
        mu, logvar = self.fc_mu(x), self.fc_var(x)
        self.mu, self.logvar = self.last(mu), self.last(logvar)
        z = utils.sample(mu, logvar)
        # logging.debug('SupRVAE Encoder final output shape {}'.format(z.shape))
        return z

    def get_instance_args(self):
        mu, logvar = self.mu, self.logvar
        del self.mu, self.logvar
        return mu, logvar


class ConvParDecoder(nn.Module):
    conv_block_mapping = {
        1: ShortConvBlock,
        2: ConvBlock,
        3: LongConvBlock,
    }

    def __init__(self, img_shape, latent_size=100, depth=3, end_filter=8, start_pixels=32, activation=utils.Mish, activation_args=None, last=nn.Tanh, residual=True, conv_block_length=2):
        super(ConvParDecoder, self).__init__()
        logging.debug('ConvDecoder parameters: {}\n'.format({
            'img_shape': img_shape,
            'latent_size': latent_size,
            'depth': depth,
            'end_filter': end_filter,
            'start_pixels': start_pixels,
        }))
        if activation_args is None:
            activation_args = {}
        assert conv_block_length in ConvParDecoder.conv_block_mapping.keys(), 'Convblock length must be a known length'
        conv_block = ConvParDecoder.conv_block_mapping[conv_block_length]

        self.img_shape = img_shape
        self.latent_size = latent_size
        self.input_shape = latent_size
        self.pixels = int(np.prod(self.img_shape))

        # decoder
        self.fc_z = nn.Linear(latent_size, self.pixels)

        self.depth = depth
        self.min_pixels = start_pixels
        self.start_pixels = start_pixels ** 2

        self.depth = depth
        current_units = end_filter

        self.blocks = []
        self.blocks.append(ResUpBlock(current_units, img_shape[0], activation=activation, activation_args=activation_args, shortcut=residual, conv_block=conv_block))
        logging.debug('Decoder level {} | In {} Out {}'.format(depth, current_units, img_shape[0]))

        for i in reversed(range(depth - 2)):
            previous_units = current_units
            current_units = current_units * 2
            self.blocks.append(ResUpBlock(current_units, previous_units, activation=activation, activation_args=activation_args, shortcut=residual, conv_block=conv_block))
            logging.debug('Decoder level {} | In {} Out {}'.format(i + 1, current_units, previous_units))

        self.max_filter = current_units * 2
        self.start_res = int(self.pixels / self.start_pixels / self.max_filter)
        self.blocks.append(ResUpBlock(self.start_res * self.max_filter, current_units, activation=activation, activation_args=activation_args, shortcut=residual, conv_block=conv_block))
        logging.debug('Decoder level {} | In {} Out {}'.format(0, self.start_res * self.max_filter, current_units))

        self.blocks = nn.ModuleList(reversed(self.blocks))
        logging.debug('Decoder Modules {} '.format(list(self.modules())))

        self.last = last()

        utils.init_weights(self.modules())

    def forward(self, z: torch.Tensor):
        # logging.debug('\nSupRVAE Decoder Input shape {}'.format(z.shape))
        z = self.fc_z(z)
        z = z.view(-1, self.start_res * self.max_filter, self.min_pixels, self.min_pixels)
        for i in range(self.depth):
            # logging.debug('SupRVAE Decoder level {} | Input shape {}'.format(i, z.shape))
            z = self.blocks[i](z)
        # logging.debug('SupRVAE Decoder level {}  | Input shape {}'.format(self.depth, z.shape))

        z = self.last(z)
        # logging.debug('SupRVAE Decoder final output shape {}'.format(z.shape))
        return z


class ConvResEncoder(nn.Module):
    def __init__(self, img_shape, latent_size=100, depth=3, start_filter=16, activation=utils.Mish, activation_args=None, residual=True, last=None, last_args=None):
        super(ConvResEncoder, self).__init__()
        logging.debug('ConvEncoder parameters: {}\n'.format({
            'img_shape': img_shape,
            'latent_size': latent_size,
            'depth': depth,
            'start_filter': start_filter,
        }))
        if activation_args is None:
            activation_args = {}
        if last_args is None:
            last_args = {}
        if last is None:
            last = activation
        self.img_shape = img_shape
        self.input_shape = img_shape
        self.latent_size = latent_size
        self.pixels = int(np.prod(self.img_shape))

        self.depth = depth

        current_units = start_filter
        # encoder

        self.first = nn.Conv2d(img_shape[0], current_units, kernel_size=3, stride=1)

        self.up = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        self.down = lambda x: F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)

        self.blocks = []
        self.blocks.append(ResDownBlock(current_units, current_units, activation=activation, activation_args=activation_args, shortcut=residual))
        logging.debug('Encoder level {} | In {} Out {}'.format(0, img_shape[0], current_units))
        for i in range(depth - 1):
            previous_units = current_units
            current_units = current_units * 2
            self.blocks.append(ResDownBlock(previous_units, current_units, activation=activation, activation_args=activation_args, shortcut=residual))
            logging.debug('Encoder level {} | In {} Out {}'.format(i + 1, previous_units, current_units))

        self.long_skip = nn.Conv2d(start_filter, current_units, kernel_size=1, stride=2**(depth-1), bias=False)
        self.blocks = nn.ModuleList(self.blocks)
        logging.debug('Encoder Modules {} '.format(list(self.modules())))

        self.middle = activation(**activation_args)

        self.fc_mu = nn.Linear(self.pixels, latent_size)
        self.fc_var = nn.Linear(self.pixels, latent_size)
        self.last = last(**last_args)
        utils.init_weights(self.modules())
        self.mu, self.logvar = None, None

    def forward(self, x: torch.Tensor):
        x = self.first(x)
        z = self.long_skip(x)
        z = self.down(z)
        # logging.debug('\nSupRVAE Encoder Input shape {}'.format(x.shape))
        for i in range(self.depth):
            # logging.debug('SupRVAE Encoder level {} | Input shape {}'.format(i, x.shape))
            x = self.blocks[i](x)
        # logging.debug('SupRVAE Encoder level {} | Input shape {}'.format(self.depth, x.shape))
        x = torch.add(x, z)
        x = x.view(-1, self.pixels)
        x = self.middle(x)
        mu, logvar = self.fc_mu(x), self.fc_var(x)
        mu, logvar = self.last(mu), self.last(logvar)
        self.mu, self.logvar = mu, logvar
        z = utils.sample(mu, logvar)
        # logging.debug('SupRVAE Encoder final output shape {}'.format(z.shape))
        return z


class ConvResDecoder(nn.Module):
    def __init__(self, img_shape, latent_size=100, depth=3, start_filter=8, start_pixels=32, activation=utils.Mish, activation_args=None, last=None, residual=True):
        super(ConvResDecoder, self).__init__()
        logging.debug('ConvResDecoder parameters: {}\n'.format({
            'img_shape': img_shape,
            'latent_size': latent_size,
            'depth': depth,
            'start_filter': start_filter,
            'start_pixels': start_pixels,
        }))
        if activation_args is None:
            activation_args = {}
        if last is None:
            last = activation

        self.img_shape = img_shape
        self.latent_size = latent_size
        self.input_shape = latent_size
        self.pixels = int(np.prod(self.img_shape))

        # decoder
        self.fc_z = nn.Linear(latent_size, self.pixels)

        self.depth = depth
        self.min_pixels = start_pixels
        self.start_pixels = start_pixels ** 2
        self.start_res = int(self.pixels / self.start_pixels)

        self.depth = depth
        current_units = start_filter

        self.blocks = []

        end_filter = start_filter * 2 ** (depth - 2)
        self.blocks.append(ResUpBlock(self.start_res, end_filter, activation=activation, activation_args=activation_args, shortcut=residual))
        logging.debug('Decoder level {} | In {} Out {}'.format(0, self.start_res, end_filter))

        for i in reversed(range(depth - 2)):
            previous_units = current_units
            current_units = current_units * 2
            self.blocks.append(ResUpBlock(current_units, previous_units, activation=activation, activation_args=activation_args, shortcut=residual))
            logging.debug('Decoder level {} | In {} Out {}'.format(i + 1, current_units, previous_units))

        self.blocks.append(ResUpBlock(start_filter, img_shape[0], activation=activation, activation_args=activation_args, shortcut=residual))
        logging.debug('Decoder level {} | In {} Out {}'.format(depth, start_filter, img_shape[0]))

        self.blocks = nn.ModuleList(self.blocks)
        self.long_skip = nn.Conv2d(self.start_res, img_shape[0], kernel_size=1, stride=1, bias=False)

        logging.debug('Decoder Modules {} '.format(list(self.modules())))

        self.last = last()
        self.up = lambda x: F.interpolate(x, scale_factor=2**(depth), mode='bilinear', align_corners=True)

        utils.init_weights(self.modules())

    def forward(self, z: torch.Tensor):
        # logging.debug('\nSupRVAE Decoder Input shape {}'.format(z.shape))
        z = self.fc_z(z)
        z = z.view(-1, self.start_res, self.min_pixels, self.min_pixels)
        x = self.long_skip(z)
        x = self.up(x)
        for i in range(self.depth):
            # logging.debug('SupRVAE Decoder level {} | Input shape {}'.format(i, z.shape))
            z = self.blocks[i](z)
        # logging.debug('SupRVAE Decoder level {}  | Input shape {}'.format(self.depth, z.shape))
        z = torch.add(x, z)

        z = self.last(z)
        # logging.debug('SupRVAE Decoder final output shape {}'.format(z.shape))
        return z


class LinearBlock(nn.Module):
    def __init__(self, in_size, out_size, activation=utils.Mish, activation_args=None, dropout=None):
        super(LinearBlock, self).__init__()
        if activation_args is None:
            activation_args = {}
        self.bn = nn.BatchNorm1d(in_size)
        self.linear = nn.Linear(in_size, out_size)
        self.act = activation(**activation_args)
        self.dropout = None
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)

    def forward(self, z):
        z = self.bn(z)
        z = self.act(z)
        z = self.linear(z)
        if self.dropout is not None:
            z = self.dropout(z)
        return z


class Discriminator(nn.Module):
    def __init__(self, latent_size=100, depth=2, start_units=128, activation=utils.Mish, activation_args=None, dropout=None):
        super(Discriminator, self).__init__()
        logging.debug('Discriminator parameters: {}\n'.format({
            'latent_size': latent_size,
            'depth': depth,
            'start_units': start_units,
        }))
        if activation_args is None:
            activation_args = {}

        self.latent_size = latent_size
        self.input_shape = latent_size
        self.depth = depth
        self.blocks = []

        current_units = start_units
        previous_units = 1
        self.blocks.append(LinearBlock(current_units, previous_units, activation=activation, activation_args=activation_args, dropout=dropout))
        logging.debug('Discriminator level {} | In {} Out {}'.format(depth, current_units, previous_units))

        for i in range(depth - 2):
            previous_units = current_units
            current_units = current_units * 2
            self.blocks.append(LinearBlock(current_units, previous_units, activation=activation, activation_args=activation_args, dropout=dropout))
            logging.debug('Discriminator level {} | In {} Out {}'.format(depth - 2 - i, current_units, previous_units))

        self.blocks.append(LinearBlock(self.latent_size, current_units, activation=activation, activation_args=activation_args, dropout=dropout))
        logging.debug('Discriminator level {} | In {} Out {}'.format(0, self.latent_size, current_units))

        self.blocks = nn.ModuleList(reversed(self.blocks))
        logging.debug('Discriminator Modules {} '.format(list(self.modules())))

        self.last = nn.Sigmoid()

        utils.init_weights(self.modules())

    def forward(self, z):
        # logging.debug('\nSupRVAE Discriminator Input shape {}'.format(z.shape))
        for i in range(self.depth):
            # logging.debug('SupRVAE Discriminator level {} | Input shape {}'.format(i, z.shape))
            z = self.blocks[i](z)
        # logging.debug('SupRVAE Discriminator level {} | Input shape {}'.format(self.depth, z.shape))

        z = self.last(z)
        # logging.debug('SupRVAE Discriminator final output shape {}'.format(z.shape))
        return z

# GAN


class ConvGenerator(nn.Module):
    def __init__(self, img_shape, latent_size=100, depth=3, end_filter=8, start_pixels=32, activation=utils.Mish, activation_args=None, last=nn.Tanh, residual=True):
        super(ConvGenerator, self).__init__()
        logging.debug('ConvGenerator parameters: {}\n'.format({
            'img_shape': img_shape,
            'latent_size': latent_size,
            'depth': depth,
            'end_filter': end_filter,
            'start_pixels': start_pixels,
        }))
        if activation_args is None:
            activation_args = {}

        self.img_shape = img_shape
        self.latent_size = latent_size
        self.input_shape = latent_size
        self.pixels = int(np.prod(self.img_shape))

        # decoder
        self.fc_z = nn.Linear(latent_size, self.pixels)

        self.depth = depth
        self.min_pixels = start_pixels
        self.start_pixels = start_pixels ** 2

        self.depth = depth
        current_units = end_filter

        self.blocks = []
        self.blocks.append(ResUpBlock(current_units, img_shape[0], activation=activation, activation_args=activation_args, shortcut=residual))
        logging.debug('Decoder level {} | In {} Out {}'.format(depth, current_units, img_shape[0]))

        for i in reversed(range(depth - 2)):
            previous_units = current_units
            current_units = current_units * 2
            self.blocks.append(ResUpBlock(current_units, previous_units, activation=activation, activation_args=activation_args, shortcut=residual))
            logging.debug('Decoder level {} | In {} Out {}'.format(i + 1, current_units, previous_units))

        self.max_filter = current_units * 2
        self.start_res = int(self.pixels / self.start_pixels / self.max_filter)
        self.blocks.append(ResUpBlock(self.start_res * self.max_filter, current_units, activation=activation, activation_args=activation_args, shortcut=residual))
        logging.debug('Decoder level {} | In {} Out {}'.format(0, self.start_res * self.max_filter, current_units))

        self.blocks = nn.ModuleList(reversed(self.blocks))
        logging.debug('Decoder Modules {} '.format(list(self.modules())))

        self.last = last()

        utils.init_weights(self.modules())

    def forward(self, z: torch.Tensor):
        # logging.debug('\nSupRVAE Decoder Input shape {}'.format(z.shape))
        z = self.fc_z(z)
        z = z.view(-1, self.start_res * self.max_filter, self.min_pixels, self.min_pixels)
        for i in range(self.depth):
            # logging.debug('SupRVAE Decoder level {} | Input shape {}'.format(i, z.shape))
            z = self.blocks[i](z)
        # logging.debug('SupRVAE Decoder level {}  | Input shape {}'.format(self.depth, z.shape))

        z = self.last(z)
        # logging.debug('SupRVAE Decoder final output shape {}'.format(z.shape))
        return z


class ConvImprinter(nn.Module):
    def __init__(self, img_shape, latent_size=100, depth=3, start_filter=16, activation=utils.Mish, activation_args=None, last=None, residual=True):
        super(ConvImprinter, self).__init__()
        logging.debug('ConvImprinter parameters: {}\n'.format({
            'img_shape': img_shape,
            'latent_size': latent_size,
            'depth': depth,
            'start_filter': start_filter,
        }))
        if activation_args is None:
            activation_args = {}
        if last is None:
            last = activation
        self.img_shape = img_shape
        self.input_shape = img_shape
        self.latent_size = latent_size
        self.pixels = int(np.prod(self.img_shape))

        self.depth = depth

        current_units = start_filter
        # encoder

        self.first = nn.Conv2d(img_shape[0], current_units, 3, stride=1)

        self.blocks = []
        self.blocks.append(ResDownBlock(current_units, current_units, activation=activation, activation_args=activation_args, shortcut=residual))
        logging.debug('Encoder level {} | In {} Out {}'.format(0, img_shape[0], current_units))

        for i in range(depth - 1):
            previous_units = current_units
            current_units = current_units * 2
            self.blocks.append(ResDownBlock(previous_units, current_units, activation=activation, activation_args=activation_args, shortcut=residual))
            logging.debug('Encoder level {} | In {} Out {}'.format(i + 1, previous_units, current_units))

        self.blocks = nn.ModuleList(self.blocks)
        logging.debug('Encoder Modules {} '.format(list(self.modules())))

        self.middle = activation(**activation_args)

        self.fc_mu = nn.Linear(self.pixels, latent_size)
        self.fc_var = nn.Linear(self.pixels, latent_size)
        self.last = last()
        utils.init_weights(self.modules())

    def forward(self, x: torch.Tensor):
        x = self.first(x)
        # logging.debug('\nSupRVAE Encoder Input shape {}'.format(x.shape))
        for i in range(self.depth):
            # logging.debug('SupRVAE Encoder level {} | Input shape {}'.format(i, x.shape))
            x = self.blocks[i](x)
        # logging.debug('SupRVAE Encoder level {} | Input shape {}'.format(self.depth, x.shape))
        x = x.view(-1, self.pixels)
        x = self.middle(x)
        mu, logvar = self.fc_mu(x), self.fc_var(x)
        mu, logvar = self.last(mu), self.last(logvar)
        z = utils.sample(mu, logvar)
        # logging.debug('SupRVAE Encoder final output shape {}'.format(z.shape))
        return z


class ConvDiscriminator(nn.Module):
    def __init__(self, img_shape, depth=3, start_filter=16, activation=utils.Mish, activation_args=None, residual=True, dropout=None):
        super(ConvDiscriminator, self).__init__()
        logging.debug('ConvDiscriminator parameters: {}\n'.format({
            'img_shape': img_shape,
            'depth': depth,
            'start_filter': start_filter,
        }))
        if activation_args is None:
            activation_args = {}
        self.img_shape = img_shape
        self.input_shape = img_shape
        self.pixels = int(np.prod(self.img_shape))

        self.depth = depth

        current_units = start_filter
        # encoder

        self.first = nn.Conv2d(img_shape[0], current_units, 3, stride=1)

        self.conv_blocks = []
        self.conv_blocks.append(ResDownBlock(current_units, current_units, activation=activation, activation_args=activation_args, shortcut=residual))
        logging.debug('Encoder level {} | In {} Out {}'.format(0, img_shape[0], current_units))

        for i in range(depth - 1):
            previous_units = current_units
            current_units = current_units * 2
            self.conv_blocks.append(ResDownBlock(previous_units, current_units, activation=activation, activation_args=activation_args, shortcut=residual))
            logging.debug('Encoder level {} | In {} Out {}'.format(i + 1, previous_units, current_units))

        self.conv_blocks = nn.ModuleList(self.conv_blocks)
        logging.debug('Encoder Modules {} '.format(list(self.modules())))

        self.middle = activation(**activation_args)

        self.linear_blocks = []
        previous_units = 1
        self.linear_blocks.append(LinearBlock(current_units, previous_units, activation=activation, activation_args=activation_args, dropout=dropout))
        logging.debug('Discriminator level {} | In {} Out {}'.format(depth, current_units, previous_units))

        for i in range(depth - 2):
            previous_units = current_units
            current_units = current_units * 2
            self.linear_blocks.append(LinearBlock(current_units, previous_units, activation=activation, activation_args=activation_args, dropout=dropout))
            logging.debug('Discriminator level {} | In {} Out {}'.format(depth - 2 - i, current_units, previous_units))

        self.linear_blocks.append(LinearBlock(self.pixels, current_units, activation=activation, activation_args=activation_args, dropout=dropout))
        logging.debug('Discriminator level {} | In {} Out {}'.format(0, self.pixels, current_units))

        self.linear_blocks = nn.ModuleList(reversed(self.linear_blocks))
        logging.debug('Discriminator Modules {} '.format(list(self.modules())))

        self.last = nn.Sigmoid()

        utils.init_weights(self.modules())

    def forward(self, x: torch.Tensor):
        x = self.first(x)
        # logging.debug('\nSupRVAE Encoder Input shape {}'.format(x.shape))
        for i in range(self.depth):
            # logging.debug('SupRVAE Encoder level {} | Input shape {}'.format(i, x.shape))
            x = self.conv_blocks[i](x)
        # logging.debug('SupRVAE Encoder level {} | Input shape {}'.format(self.depth, x.shape))
        x = x.view(-1, self.pixels)
        x = self.middle(x)
        for i in range(self.depth):
            # logging.debug('SupRVAE Discriminator level {} | Input shape {}'.format(i, z.shape))
            x = self.linear_blocks[i](x)
        # logging.debug('SupRVAE Discriminator level {} | Input shape {}'.format(self.depth, z.shape))

        x = self.last(x)
        # logging.debug('SupRVAE Encoder final output shape {}'.format(z.shape))
        return x


