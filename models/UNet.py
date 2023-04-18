# Adapted from https://discuss.pytorch.org/t/unet-implementation/426

import torch
import math
from torch import nn
import torch.nn.functional as F
import models.utils as utils


def create_conv_block(in_size, out_size, padding, batch_norm, activation=utils.Mish):
    block = [nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)),
             activation(),
             nn.BatchNorm2d(out_size) if batch_norm else None,
             nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)),
             activation(),
             nn.BatchNorm2d(out_size) if batch_norm else None,
             ]

    block = filter(lambda layer: layer is not None, block)

    return nn.Sequential(*block)


class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size,  padding, batch_norm, activation=utils.Mish):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding))
        self.act1 = activation()
        self.bn1 = nn.BatchNorm2d(in_size) if batch_norm else None
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding))
        self.act2 = activation()
        self.bn2 = nn.BatchNorm2d(out_size) if batch_norm else None
        # logging.info('{} -> {} | {}'.format(in_size, out_size, stride))

    def forward(self, x):
        # # logging.info('conv block:')
        # prev_shape = x.shape
        x = self.conv1(x)
        # # logging.info('conv1:\t{}x{}x{}x{}\t->\t{}x{}x{}x{}'.format(*prev_shape, *x.shape))
        # prev_shape = x.shape
        x = self.act1(x)
        # # logging.info('act1:\t{}x{}x{}x{}\t->\t{}x{}x{}x{}'.format(*prev_shape, *x.shape))
        # prev_shape = x.shape
        x = self.bn1(x) if self.bn1 is not None else x
        # # logging.info('bn1:\t{}x{}x{}x{}\t->\t{}x{}x{}x{}'.format(*prev_shape, *x.shape))
        # prev_shape = x.shape
        x = self.conv2(x)
        # # logging.info('conv2:\t{}x{}x{}x{}\t->\t{}x{}x{}x{}'.format(*prev_shape, *x.shape))
        # prev_shape = x.shape
        x = self.act2(x)
        # # logging.info('act2:\t{}x{}x{}x{}\t->\t{}x{}x{}x{}'.format(*prev_shape, *x.shape))
        # prev_shape = x.shape
        x = self.bn2(x) if self.bn2 is not None else x
        # # logging.info('bn2:\t{}x{}x{}x{}\t->\t{}x{}x{}x{}'.format(*prev_shape, *x.shape))
        # log_shapes('conv_block', prev_shape, x.shape)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, depth=5, wf=6, padding=False, batch_norm=False, up_mode='upconv', activation=utils.Mish, last=utils.Mish):
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        self.wf = wf
        self.batch_norm = batch_norm
        self.padding = padding
        self.up_mode = up_mode
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetDownBlock(prev_channels, 2 ** (wf + i), padding, batch_norm, activation=activation))
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2 ** (wf + i), padding, batch_norm, up_mode, activation=activation))
            prev_channels = 2 ** (wf + i)
        self.up_path.append(nn.Conv2d(prev_channels, n_classes, kernel_size=1))

        self.last = last()

        utils.init_weights(self.modules())

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class UNetDownBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm, activation=utils.Mish):
        super(UNetDownBlock, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.block = create_conv_block(in_size, out_size, padding, batch_norm, activation=activation)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm, up_mode, activation=utils.Mish):
        super(UNetUpBlock, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,
                                         stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=1))
        self.conv_block = create_conv_block(in_size, out_size, padding, batch_norm, activation=activation)

    @staticmethod
    def center_crop(layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out
