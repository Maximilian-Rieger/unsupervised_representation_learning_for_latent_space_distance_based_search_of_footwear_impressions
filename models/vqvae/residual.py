
import torch
import torch.nn as nn
import numpy as np


class ResidualLayer(nn.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """

    def __init__(self, in_dim, h_dim, res_h_dim, padding_mode='zeros', batch_norm=False):
        super(ResidualLayer, self).__init__()
        self.res_block = []
        if batch_norm:
            self.res_block += [nn.BatchNorm2d(in_dim)]
        self.res_block += [
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3, stride=1, padding=1, bias=False, padding_mode=padding_mode),
        ]
        if batch_norm:
            self.res_block += [nn.BatchNorm2d(res_h_dim)]
        self.res_block += [
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1, stride=1, bias=False, padding_mode=padding_mode)
            ]
        self.res_block = nn.Sequential(*self.res_block)

    def forward(self, x):
        x = x + self.res_block(x)
        return x


class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers, batch_norm=False):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.batch_norm = batch_norm
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim, batch_norm=batch_norm)]*n_res_layers)
        if batch_norm:
            self.bn = nn.BatchNorm2d(in_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        if self.batch_norm:
            x = self.bn(x)
        x = self.act(x)
        return x


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 40, 40, 200))
    x = torch.tensor(x).float()
    # test Residual Layer
    res = ResidualLayer(40, 40, 20)
    res_out = res(x)
    print('Res Layer out shape:', res_out.shape)
    # test res stack
    res_stack = ResidualStack(40, 40, 20, 3)
    res_stack_out = res_stack(x)
    print('Res Stack out shape:', res_stack_out.shape)
