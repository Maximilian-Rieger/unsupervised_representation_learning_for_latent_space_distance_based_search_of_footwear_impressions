
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.vqvae.residual import ResidualStack


class RasterEncoder(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, batch_norm=False):
        super(RasterEncoder, self).__init__()
        kernel = 4
        stride = 2
        self.in_dim = in_dim
        self.h_dim = h_dim // 2
        self.conv_stack = []
        self.conv_stack.append(
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
        )
        self.conv_stack.append(
            nn.ReLU()
        )
        if batch_norm:
            self.conv_stack.append(
                nn.BatchNorm2d(h_dim // 2)
            )
        self.conv_stack.append(
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1),
        )
        self.conv_stack.append(
            nn.ReLU()
        )
        if batch_norm:
            self.conv_stack.append(
                nn.BatchNorm2d(h_dim)
            )
        self.conv_stack.append(
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),
        )
        self.conv_stack.append(
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers, batch_norm=batch_norm)
        )
        self.conv_stack = nn.Sequential(*self.conv_stack)

    def forward(self, x):
        return self.conv_stack(x)


class RasterEncoder_b(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, batch_norm=False):
        super(RasterEncoder_b, self).__init__()
        kernel = 4
        stride = 2
        self.in_dim = in_dim
        self.h_dim = h_dim // 2
        self.conv_stack = []
        self.conv_stack.append(
            nn.Conv2d(in_dim, h_dim // 4, kernel_size=kernel*2, stride=stride, padding=4),
        )
        self.conv_stack.append(
            nn.ReLU()
        )
        if batch_norm:
            self.conv_stack.append(
                nn.BatchNorm2d(h_dim // 4)
            )
        self.conv_stack.append(
            nn.Conv2d(h_dim // 4, h_dim // 4, kernel_size=kernel, stride=stride, padding=1),
        )
        self.conv_stack.append(
            nn.ReLU()
        )
        if batch_norm:
            self.conv_stack.append(
                nn.BatchNorm2d(h_dim // 4)
            )
        self.conv_stack.append(
            nn.Conv2d(h_dim // 4, h_dim // 2, kernel_size=kernel, stride=stride, padding=1, padding_mode='replicate'),
        )
        self.conv_stack.append(
            nn.ReLU()
        )
        if batch_norm:
            self.conv_stack.append(
                nn.BatchNorm2d(h_dim // 2)
            )
        self.conv_stack.append(
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1, padding_mode='replicate'),
        )
        self.conv_stack.append(
            nn.ReLU()
        )
        if batch_norm:
            self.conv_stack.append(
                nn.BatchNorm2d(h_dim)
            )
        self.conv_stack.append(
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1, padding_mode='replicate'),
        )
        self.conv_stack.append(
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers, batch_norm=batch_norm)
        )
        self.conv_stack = nn.Sequential(*self.conv_stack)

    def forward(self, x):
        return self.conv_stack(x)


class RasterEncoder_2(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, batch_norm=False):
        super(RasterEncoder_2, self).__init__()
        kernel = 4
        stride = 2
        self.in_dim = in_dim
        self.h_dim = h_dim // 2
        self.conv_stack = []
        self.conv_stack += [
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel * 2, stride=stride, padding=1+4, padding_mode='replicate'),
            nn.ReLU()
        ]
        if batch_norm:
            self.conv_stack += [nn.BatchNorm2d(h_dim // 2)]
        self.conv_stack += [
            nn.Conv2d(h_dim // 2, h_dim // 2, kernel_size=kernel*2, stride=stride, padding=1+4, padding_mode='replicate'),
            nn.ReLU()
        ]
        if batch_norm:
            self.conv_stack += [nn.BatchNorm2d(h_dim // 2)]
        self.conv_stack += [
            nn.Conv2d(h_dim // 2, h_dim // 2, kernel_size=kernel, stride=stride, padding=1, padding_mode='replicate'),
            nn.ReLU()
        ]
        if batch_norm:
            self.conv_stack += [nn.BatchNorm2d(h_dim // 2)]
        self.conv_stack += [
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1, padding_mode='replicate'),
            nn.ReLU()
        ]
        if batch_norm:
            self.conv_stack += [nn.BatchNorm2d(h_dim)]
        self.conv_stack += [
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1, padding_mode='replicate'),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers, batch_norm=batch_norm)
        ]

        self.conv_stack = nn.Sequential(*self.conv_stack)

    def forward(self, x):
        return self.conv_stack(x)


class RasterEncoder_3(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, batch_norm=False):
        super(RasterEncoder_3, self).__init__()
        kernel = 4
        stride = 2
        self.in_dim = in_dim
        self.h_dim = h_dim // 2
        self.conv_stack = []
        self.conv_stack += [
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel * 2, stride=stride, padding=1+4, padding_mode='replicate'),
            nn.ReLU()
        ]
        if batch_norm:
            self.conv_stack += [nn.BatchNorm2d(h_dim // 2)]
        self.conv_stack += [
            nn.Conv2d(h_dim // 2, h_dim // 2, kernel_size=kernel*2, stride=stride, padding=1+4, padding_mode='replicate'),
            nn.ReLU()
        ]
        if batch_norm:
            self.conv_stack += [nn.BatchNorm2d(h_dim // 2)]
        self.conv_stack += [
            nn.Conv2d(h_dim // 2, h_dim // 2, kernel_size=kernel, stride=stride, padding=1, padding_mode='replicate'),
            nn.ReLU()
        ]
        if batch_norm:
            self.conv_stack += [nn.BatchNorm2d(h_dim // 2)]
        self.conv_stack_1 = nn.Sequential(*self.conv_stack)
        self.max_pool = nn.MaxPool2d(kernel_size=kernel, stride=stride, padding=1, return_indices=True)
        self.max_pool_act = nn.ReLU()
        # self.conv_stack += [
        #     nn.MaxPool2d(kernel_size=kernel, stride=stride, padding=1, return_indices=True),
        #     nn.ReLU()
        # ]
        self.conv_stack = []
        if batch_norm:
            self.conv_stack += [nn.BatchNorm2d(h_dim // 2)]
        self.conv_stack += [
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1, padding_mode='replicate'),
            nn.ReLU()
        ]
        if batch_norm:
            self.conv_stack += [nn.BatchNorm2d(h_dim)]
        self.conv_stack += [
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1, padding_mode='replicate'),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers, batch_norm=batch_norm)
        ]

        self.conv_stack_2 = nn.Sequential(*self.conv_stack)

    def forward(self, x):
        x = self.conv_stack_1(x)
        x, indices = self.max_pool(x)
        x = self.max_pool_act(x)
        x = self.conv_stack_2(x)
        return x, indices


class RasterEncoder_4(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, batch_norm=False):
        super(RasterEncoder_4, self).__init__()
        kernel = 4
        stride = 2
        self.in_dim = in_dim
        self.h_dim = h_dim // 2
        self.conv_stack = []
        self.conv_stack += [
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel * 2, stride=stride, padding=1+4, padding_mode='replicate'),
            nn.ReLU()
        ]
        if batch_norm:
            self.conv_stack += [nn.BatchNorm2d(h_dim // 2)]
        self.conv_stack += [
            nn.Conv2d(h_dim // 2, h_dim // 2, kernel_size=kernel*2, stride=stride, padding=1+4, padding_mode='replicate', dilation=2),
            nn.ReLU()
        ]
        if batch_norm:
            self.conv_stack += [nn.BatchNorm2d(h_dim // 2)]
        self.conv_stack += [
            nn.Conv2d(h_dim // 2, h_dim // 2, kernel_size=kernel, stride=stride, padding=1, padding_mode='replicate', dilation=2),
            nn.ReLU()
        ]
        if batch_norm:
            self.conv_stack += [nn.BatchNorm2d(h_dim // 2)]
        self.conv_stack += [
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1, padding_mode='replicate'),
            nn.ReLU()
        ]
        if batch_norm:
            self.conv_stack += [nn.BatchNorm2d(h_dim)]
        self.conv_stack += [
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1, padding_mode='replicate'),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers, batch_norm=batch_norm)
        ]

        self.conv_stack = nn.Sequential(*self.conv_stack)

    def forward(self, x):
        return self.conv_stack(x)


class GeneralRasterEncoder(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, down_scale=2, start_kernel=4, down_kernel=4, batch_norm=False, padding_mode='zeros'):
        super(GeneralRasterEncoder, self).__init__()
        kernel = start_kernel
        stride = 2
        self.in_dim = in_dim
        self.h_dim = h_dim // 2
        self.down_scale = down_scale

        self.conv_stack = []
        self.conv_stack += [
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1+(start_kernel-4), padding_mode=padding_mode),
            nn.ReLU(),
        ]

        if batch_norm:
            self.conv_stack += [nn.BatchNorm2d(h_dim // 2)]
        self.conv_stack += [
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1+(start_kernel-4), padding_mode=padding_mode),
            nn.ReLU(),
        ]
        if batch_norm:
            self.conv_stack += [nn.BatchNorm2d(h_dim)]
        if down_scale > 2:
            for n in range(3, down_scale + 1):
                self.conv_stack += [
                    nn.Conv2d(h_dim, h_dim, kernel_size=down_kernel, stride=stride, padding=1+(down_kernel-kernel)+(start_kernel-4), padding_mode=padding_mode),
                    nn.ReLU(),
                ]
                if batch_norm:
                    self.conv_stack += [nn.BatchNorm2d(h_dim)]

        self.conv_stack += [
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1+(start_kernel-4)),
        ]
        self.conv_stack += [ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers, batch_norm=batch_norm)]
        self.conv_stack = nn.Sequential(*self.conv_stack)

    def forward(self, x):
        return self.conv_stack(x)


class Encoder(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, col_kernel=64, batch_norm=False):
        super(Encoder, self).__init__()
        kernel = 4
        stride = 2
        self.in_dim = in_dim
        self.h_dim = h_dim // 2
        self.conv1 = nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1)
        self.act1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(h_dim // 2) if batch_norm else nn.Identity()

        self.conv2 = nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1)
        self.act2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(h_dim) if batch_norm else nn.Identity()

        self.conv3 = nn.Conv2d(h_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1)
        self.rstack = ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers, batch_norm=batch_norm)

        self.col_conv = nn.Conv2d(h_dim, h_dim, kernel_size=col_kernel, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(h_dim) if batch_norm else nn.Identity()
        self.col_act = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.rstack(x)

        x = self.col_conv(x)
        x = self.bn3(x)
        x = self.col_act(x)

        return x


class RFEncoder(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, rf_scale=2, batch_norm=False):
        super(RFEncoder, self).__init__()
        kernel = 4
        stride = 2
        self.in_dim = in_dim
        self.rf_scale = rf_scale
        self.h_dim = h_dim // 2
        self.conv1 = nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1)
        self.act1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(h_dim // 2) if batch_norm else nn.Identity()

        self.conv2 = nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1)
        self.act2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(h_dim) if batch_norm else nn.Identity()

        self.conv3 = nn.Conv2d(h_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1)
        self.rstack = ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers, batch_norm=batch_norm)
        self.bn3 = nn.BatchNorm2d(h_dim) if batch_norm else nn.Identity()

        # self.col_conv = nn.Conv2d(h_dim, h_dim, kernel_size=col_kernel//stride, stride=stride, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.rstack(x)
        x = self.bn3(x)

        # x = self.col_conv(x)

        return x


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 40, 40, 200))
    x = torch.tensor(x).float()

    # test encoder
    encoder = RasterEncoder(40, 128, 3, 64)
    encoder_out = encoder(x)
    print('RasterEncoder out shape:', encoder_out.shape)
