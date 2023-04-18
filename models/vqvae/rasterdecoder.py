
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.vqvae.residual import ResidualStack


class RasterDecoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, out_chan, batch_norm=False):
        super(RasterDecoder, self).__init__()
        kernel = 4
        stride = 2
        self.in_dim = in_dim
        self.h_dim = h_dim

        self.inverse_conv_stack = []
        self.inverse_conv_stack.append(
            nn.ConvTranspose2d(in_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1)
        )
        self.inverse_conv_stack.append(
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers, batch_norm=batch_norm)
        )
        self.inverse_conv_stack.append(
            nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1)
        )
        self.inverse_conv_stack.append(
            nn.ReLU()
        )
        if batch_norm:
            self.inverse_conv_stack.append(
                nn.BatchNorm2d(h_dim // 2)
            )
        self.inverse_conv_stack.append(
            nn.ConvTranspose2d(h_dim//2, out_chan, kernel_size=kernel,stride=stride, padding=1)
        )

        self.inverse_conv_stack = nn.Sequential(*self.inverse_conv_stack)

    def forward(self, x):
        return self.inverse_conv_stack(x)


class RasterDecoder_b(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, out_chan, batch_norm=False):
        super(RasterDecoder_b, self).__init__()
        kernel = 4
        stride = 2
        self.in_dim = in_dim
        self.h_dim = h_dim

        self.inverse_conv_stack = []
        self.inverse_conv_stack.append(
            nn.ConvTranspose2d(in_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1)
        )
        self.inverse_conv_stack.append(
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers, batch_norm=batch_norm)
        )
        self.inverse_conv_stack.append(
            nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1)
        )
        self.inverse_conv_stack.append(
            nn.ReLU()
        )
        if batch_norm:
            self.inverse_conv_stack.append(
                nn.BatchNorm2d(h_dim // 2)
            )
        self.inverse_conv_stack.append(
            nn.ConvTranspose2d(h_dim//2, h_dim//4, kernel_size=kernel,stride=stride, padding=1)
        )
        self.inverse_conv_stack.append(
            nn.ReLU()
        )

        if batch_norm:
            self.inverse_conv_stack.append(
                nn.BatchNorm2d(h_dim // 4)
            )
        self.inverse_conv_stack.append(
            nn.ConvTranspose2d(h_dim//4, h_dim//4, kernel_size=kernel,stride=stride, padding=1)
        )
        self.inverse_conv_stack.append(
            nn.ReLU()
        )
        if batch_norm:
            self.inverse_conv_stack.append(
                nn.BatchNorm2d(h_dim // 4)
            )
        self.inverse_conv_stack.append(
            nn.ConvTranspose2d(h_dim//4, out_chan, kernel_size=kernel*2,stride=stride, padding=3)
        )

        self.inverse_conv_stack = nn.Sequential(*self.inverse_conv_stack)

    def forward(self, x):
        return self.inverse_conv_stack(x)


class RasterDecoder_2(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, out_chan, batch_norm=False):
        super(RasterDecoder_2, self).__init__()
        kernel = 4
        stride = 2
        self.in_dim = in_dim
        self.h_dim = h_dim

        self.inverse_conv_stack = []
        self.inverse_conv_stack += [
            nn.ConvTranspose2d(in_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers, batch_norm=batch_norm),
            nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU()
        ]
        if batch_norm:
            self.inverse_conv_stack += [nn.BatchNorm2d(h_dim // 2)]
        self.inverse_conv_stack += [
            nn.ConvTranspose2d(h_dim//2, h_dim//2, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU()
        ]
        if batch_norm:
            self.inverse_conv_stack += [nn.BatchNorm2d(h_dim // 2)]
        self.inverse_conv_stack += [
            nn.ConvTranspose2d(h_dim//2, h_dim//2, kernel_size=kernel * 2, stride=stride, padding=1+2),
            nn.ReLU()
        ]
        if batch_norm:
            self.inverse_conv_stack += [nn.BatchNorm2d(h_dim // 2)]
        self.inverse_conv_stack += [
            nn.ConvTranspose2d(h_dim // 2, out_chan, kernel_size=kernel * 2, stride=stride, padding=1+2)
        ]

        self.inverse_conv_stack = nn.Sequential(*self.inverse_conv_stack)

    def forward(self, x):
        return self.inverse_conv_stack(x)


class RasterDecoder_3(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, out_chan, batch_norm=False):
        super(RasterDecoder_3, self).__init__()
        kernel = 4
        stride = 2
        self.in_dim = in_dim
        self.h_dim = h_dim

        self.inverse_conv_stack = []
        self.inverse_conv_stack += [
            nn.ConvTranspose2d(in_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers, batch_norm=batch_norm),
            nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU()
        ]
        if batch_norm:
            self.inverse_conv_stack += [nn.BatchNorm2d(h_dim // 2)]
        self.inverse_conv_stack_1 = nn.Sequential(*self.inverse_conv_stack)

        self.max_unpool = nn.MaxUnpool2d(kernel_size=kernel, stride=stride, padding=1)
        self.max_unpool_act = nn.ReLU()

        self.inverse_conv_stack = []
        if batch_norm:
            self.inverse_conv_stack += [nn.BatchNorm2d(h_dim // 2)]
        self.inverse_conv_stack += [
            nn.ConvTranspose2d(h_dim//2, h_dim//2, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU()
        ]
        if batch_norm:
            self.inverse_conv_stack += [nn.BatchNorm2d(h_dim // 2)]
        self.inverse_conv_stack += [
            nn.ConvTranspose2d(h_dim//2, h_dim//2, kernel_size=kernel * 2, stride=stride, padding=1+2),
            nn.ReLU()
        ]
        if batch_norm:
            self.inverse_conv_stack += [nn.BatchNorm2d(h_dim // 2)]
        self.inverse_conv_stack += [
            nn.ConvTranspose2d(h_dim // 2, out_chan, kernel_size=kernel * 2, stride=stride, padding=1+2)
        ]

        self.inverse_conv_stack_2 = nn.Sequential(*self.inverse_conv_stack)

    def forward(self, x, indices):
        x = self.inverse_conv_stack_1(x)
        x = self.max_unpool(x, indices)
        x = self.max_unpool_act(x)
        x = self.inverse_conv_stack_2(x)
        return x




class RasterDecoder_4(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, out_chan, batch_norm=False):
        super(RasterDecoder_4, self).__init__()
        kernel = 4
        stride = 2
        self.in_dim = in_dim
        self.h_dim = h_dim

        self.inverse_conv_stack = []
        self.inverse_conv_stack += [
            nn.ConvTranspose2d(in_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers, batch_norm=batch_norm),
            nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU()
        ]
        if batch_norm:
            self.inverse_conv_stack += [nn.BatchNorm2d(h_dim // 2)]
        self.inverse_conv_stack += [
            nn.ConvTranspose2d(h_dim//2, h_dim//2, kernel_size=kernel, stride=stride, padding=1+1, dilation=2),
            nn.ReLU()
        ]
        if batch_norm:
            self.inverse_conv_stack += [nn.BatchNorm2d(h_dim // 2)]
        self.inverse_conv_stack += [
            nn.ConvTranspose2d(h_dim//2, h_dim//2, kernel_size=kernel * 2, stride=stride, padding=1+2, dilation=2),
            nn.ReLU()
        ]
        if batch_norm:
            self.inverse_conv_stack += [nn.BatchNorm2d(h_dim // 2)]
        self.inverse_conv_stack += [
            nn.ConvTranspose2d(h_dim // 2, out_chan, kernel_size=kernel * 2, stride=stride, padding=1+3)
        ]

        self.inverse_conv_stack = nn.Sequential(*self.inverse_conv_stack)

    def forward(self, x):
        return self.inverse_conv_stack(x)


class GeneralRasterDecoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, out_chan, up_scale=2, end_kernel=4, up_kernel=4, batch_norm=False, padding_mode='zeros'):
        super(GeneralRasterDecoder, self).__init__()
        kernel = end_kernel
        stride = 2
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.up_scale = up_scale

        self.inverse_conv_stack = []

        self.inverse_conv_stack += []
        self.inverse_conv_stack += [
            nn.ConvTranspose2d(in_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1+(end_kernel-4), padding_mode=padding_mode),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers, batch_norm=batch_norm),
        ]
        if up_scale > 2:
            for n in range(3, up_scale+1):
                self.inverse_conv_stack += [
                    nn.ConvTranspose2d(h_dim, h_dim, kernel_size=up_kernel, stride=stride, padding=1+(up_kernel-4), padding_mode=padding_mode),
                    nn.ReLU(),
                ]
                if batch_norm:
                    self.inverse_conv_stack += [nn.BatchNorm2d(h_dim)]

        self.inverse_conv_stack += [
            nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1+(end_kernel-4), padding_mode=padding_mode),
            nn.ReLU(),
        ]
        if batch_norm:
            self.inverse_conv_stack += [nn.BatchNorm2d(h_dim // 2)]
        self.inverse_conv_stack += [nn.ConvTranspose2d(h_dim//2, out_chan, kernel_size=kernel, stride=stride, padding=1+(end_kernel-4), padding_mode=padding_mode)]

        self.inverse_conv_stack = nn.Sequential(*self.inverse_conv_stack)

    def forward(self, x):
        return self.inverse_conv_stack(x)



class Decoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, out_chan, exp_kernel=64, batch_norm=False):
        super(Decoder, self).__init__()
        kernel = 4
        stride = 2
        self.in_dim = in_dim
        self.h_dim = h_dim

        self.exp_conv = nn.ConvTranspose2d(in_dim, in_dim, kernel_size=exp_kernel, stride=1, padding=0)
        self.act0 = nn.ReLU()
        self.bn0 = nn.BatchNorm2d(in_dim) if batch_norm else nn.Identity()

        self.iconv1 = nn.ConvTranspose2d(in_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1)
        self.rstack = ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers, batch_norm=batch_norm)
        self.iconv2 = nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1)
        self.act1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(h_dim // 2) if batch_norm else nn.Identity()
        self.iconv3 = nn.ConvTranspose2d(h_dim//2, out_chan, kernel_size=kernel, stride=stride, padding=1)


    def forward(self, x):
        x = self.exp_conv(x)
        x = self.act0(x)
        x = self.bn0(x)

        x = self.iconv1(x)
        x = self.rstack(x)
        x = self.iconv2(x)
        x = self.act1(x)
        x = self.bn1(x)
        x = self.iconv3(x)

        return x


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 40, 40, 200))
    x = torch.tensor(x).float()

    # test decoder
    decoder = RasterDecoder(40, 128, 3, 64)
    decoder_out = decoder(x)
    print('Dncoder out shape:', decoder_out.shape)
