# Adapted from https://discuss.pytorch.org/t/unet-implementation/426

import torch
import logging
from torch import nn
import torch.nn.functional as F

import models.utils as utils


class LongConvBlock(nn.Module):
    def __init__(self, in_size, out_size, stride, activation=nn.ReLU, activation_args=None):
        super(LongConvBlock, self).__init__()
        if activation_args is None:
            activation_args = {}

        self.block1 = ShortConvBlock(in_size, in_size, 1, activation, activation_args)
        self.block2 = ShortConvBlock(in_size, out_size, stride, activation, activation_args)
        self.block3 = ShortConvBlock(out_size, out_size, 1, activation, activation_args)
        logging.debug('{} -> {} | {}'.format(in_size, out_size, stride))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, stride, activation=nn.ReLU, activation_args=None):
        super(ConvBlock, self).__init__()
        if activation_args is None:
            activation_args = {}

        self.bn1 = nn.BatchNorm2d(in_size)
        self.act1 = activation(**activation_args)
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.act2 = activation(**activation_args)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        logging.debug('{} -> {} | {}'.format(in_size, out_size, stride))

    def forward(self, x):
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv2(x)
        return x


class ShortConvBlock(nn.Module):
    def __init__(self, in_size, out_size, stride, activation=nn.ReLU, activation_args=None):
        super(ShortConvBlock, self).__init__()
        if activation_args is None:
            activation_args = {}
        self.bn1 = nn.BatchNorm2d(in_size)
        self.act1 = activation(**activation_args)
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, stride=stride, padding=1)
        logging.debug('{} -> {} | {}'.format(in_size, out_size, stride))

    def forward(self, x):
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv1(x)
        return x


class ResUNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, depth=4, wf=6, activation=nn.ReLU, last=nn.Sigmoid):
        super(ResUNet, self).__init__()
        self.depth = depth
        self.wf = wf

        self.save_activations = False

        prev_channels = in_channels
        self.down_path = nn.ModuleList()

        self.down_path.append(ResUNetDownBlock(prev_channels, 2 ** (wf + 0), True, activation=activation))
        prev_channels = 2 ** (wf + 0)
        for i in range(1, depth):
            self.down_path.append(ResUNetDownBlock(prev_channels, 2 ** (wf + i)))
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(ResUNetUpBlock(prev_channels + 2 ** (wf + i), 2 ** (wf + i)))
            prev_channels = 2 ** (wf + i)

        self.last_conv = nn.Conv2d(prev_channels, n_classes, kernel_size=1)
        self.last = last()
        # logging.info('{} -> {}'.format(prev_channels, n_classes))

        utils.init_weights(self.modules())

    def forward(self, x):
        # print()
        blocks = []
        # logging.info('down:')
        for i, down in enumerate(self.down_path):
            # if not i < len(self.down_path) - 1:
                # logging.info('bridge:')
            x = down(x)
            if i < len(self.down_path) - 1:
                blocks.append(x)

        # logging.info('up:')
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])
        if self.save_activations:
            self.activations = [block.detach() for block in blocks]
        x = self.last_conv(x)
        return self.last(x)

    def add_level(self):
        prev_channels = self.down_path[self.depth - 1].out_size

        down = ResUNetDownBlock(prev_channels, 2 ** (self.wf + self.depth))
        utils.init_weights(down.modules())
        self.down_path.append(down)
        prev_channels = 2 ** (self.wf + self.depth)

        up = ResUNetUpBlock(prev_channels, 2 ** (self.wf + self.depth - 1))
        utils.init_weights(up.modules())
        self.up_path = nn.ModuleList([up, self.up_path])
        self.depth += 1
        return [down, up]

    def encoder(self):
        encoder_path = self.down_path
        return encoder_path

    def decoder(self):
        decoder_path = self.up_path + self.last_conv
        return decoder_path


class ResUNetDownBlock(nn.Module):
    def __init__(self, in_size, out_size, start_block=False, activation=nn.ReLU):
        super(ResUNetDownBlock, self).__init__()
        self.start_block = start_block
        self.in_size = in_size
        self.out_size = out_size
        if not start_block:
            self.stride = 2
            self.block = ConvBlock(in_size, out_size, self.stride, activation=activation)
        else:
            self.stride = 1
            self.block = ShortConvBlock(in_size, out_size, self.stride, activation=activation)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=1, stride=self.stride, bias=False),
            nn.BatchNorm2d(out_size),
        )

    def forward(self, x):
        out = self.block(x)
        x = self.shortcut(x)
        out = torch.add(x, out)

        return out


class ResDownBlock(nn.Module):
    def __init__(self, in_size, out_size, stride=2, conv_block=ConvBlock, activation=nn.ReLU, activation_args=None, batch_norm=False, shortcut=True):
        super(ResDownBlock, self).__init__()
        if activation_args is None:
            activation_args = {}
        self.in_size = in_size
        self.out_size = out_size
        self.stride = stride
        self.batch_norm = batch_norm
        self.block = conv_block(in_size, out_size, self.stride, activation=activation, activation_args=activation_args)
        self.shortcut_enabled = shortcut

        if self.shortcut_enabled:
            self.shortcut = [nn.Conv2d(in_size, out_size, kernel_size=1, stride=self.stride, bias=False)]
            if batch_norm:
                self.shortcut += nn.BatchNorm2d(out_size)
            self.shortcut = nn.Sequential(*self.shortcut)

    def forward(self, x):
        out = self.block(x)
        if self.shortcut_enabled:
            x = self.shortcut(x)
            out = torch.add(x, out)

        return out


class ResUNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, activation=nn.ReLU, activation_args=None):
        super(ResUNetUpBlock, self).__init__()
        if activation_args is None:
            activation_args = {}
        self.in_size = in_size
        self.out_size = out_size

        self.up = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_block = ConvBlock(in_size, out_size, 1, activation=activation, activation_args=activation_args)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_size),
        )

    def forward(self, x, bridge):
        # logging.info('x  {} bridge {}'.format(shape_description(x.shape), shape_description(bridge.shape)))
        up = self.up(x)
        x = torch.cat([up, bridge], 1)
        # logging.info('up {} concat {}'.format(shape_description(up.shape), shape_description(x.shape)))
        out = self.conv_block(x)
        x = self.shortcut(x)
        out = torch.add(x, out)

        return out


class ResUpBlock(nn.Module):
    def __init__(self, in_size, out_size, activation=nn.ReLU, conv_block=ConvBlock, activation_args=None, batch_norm=False, shortcut=True):
        super(ResUpBlock, self).__init__()
        if activation_args is None:
            activation_args = {}
        self.in_size = in_size
        self.out_size = out_size
        self.batch_norm = batch_norm

        self.up = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_block = conv_block(in_size, out_size, 1, activation=activation, activation_args=activation_args)

        self.shortcut_enabled = shortcut
        if self.shortcut_enabled:
            self.shortcut = [nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, bias=False)]
            if batch_norm:
                self.shortcut += nn.BatchNorm2d(out_size)
            self.shortcut = nn.Sequential(*self.shortcut)

    def forward(self, x):
        up = self.up(x)
        out = self.conv_block(up)
        if self.shortcut_enabled:
            x = self.shortcut(up)
            out = torch.add(x, out)

        return out

