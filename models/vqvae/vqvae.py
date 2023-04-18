
import torch
import torch.nn as nn
import numpy as np
from models.vqvae.rasterencoder import GeneralRasterEncoder, RasterEncoder, RasterEncoder_b, RasterEncoder_2, RasterEncoder_3, RasterEncoder_4, Encoder
from models.vqvae.quantizer import VectorQuantizer
from models.vqvae.rasterdecoder import GeneralRasterDecoder, RasterDecoder, RasterDecoder_b, RasterDecoder_2, RasterDecoder_3, RasterDecoder_4, Decoder


class VQVAE(nn.Module):
    def __init__(
            self,
            in_chan=3,
            h_dim=128,
            res_h_dim=32,
            n_res_layers=2,
            n_embeddings=512,
            embedding_dim=64,
            beta=0.25,
            save_img_embedding_map=False,
            batch_norm=False
    ):
        super(VQVAE, self).__init__()
        self.in_chan = in_chan
        self.embedding_dim = embedding_dim
        # encode image into continuous latent space
        self.encoder = RasterEncoder(in_chan, h_dim, n_res_layers, res_h_dim, batch_norm=batch_norm)
        self.pre_quantization_conv = nn.Conv2d(h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = RasterDecoder(embedding_dim, h_dim, n_res_layers, res_h_dim, in_chan, batch_norm=batch_norm)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def reconstruct(self, x):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        z_q, _, _ = self.vector_quantization.strait_through(z_e)
        x_hat = self.decoder(z_q)

        return x_hat, z_e, z_q

    def forward(self, x, verbose=False):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)

        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity, z_q

    def encode(self, x):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)

        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)

        return z_e, z_q

    def decode(self, z_q, quantify=False):
        if quantify:
            _, z_q, _, _ = self.vector_quantization(z_q)

        return self.decoder(z_q)


class VQVAE_b(nn.Module):
    def __init__(
            self,
            in_chan=3,
            h_dim=128,
            res_h_dim=32,
            n_res_layers=2,
            n_embeddings=512,
            embedding_dim=64,
            beta=0.25,
            save_img_embedding_map=False,
            batch_norm=False
    ):
        super(VQVAE_b, self).__init__()
        self.in_chan = in_chan
        self.embedding_dim = embedding_dim
        # encode image into continuous latent space
        self.encoder = RasterEncoder_b(in_chan, h_dim, n_res_layers, res_h_dim, batch_norm=batch_norm)
        self.pre_quantization_conv = nn.Conv2d(h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = RasterDecoder_b(embedding_dim, h_dim, n_res_layers, res_h_dim, in_chan, batch_norm=batch_norm)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def reconstruct(self, x):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        z_q, _, _ = self.vector_quantization.strait_through(z_e)
        x_hat = self.decoder(z_q)

        return x_hat, z_e, z_q

    def forward(self, x, verbose=False):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)

        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity, z_q

    def encode(self, x):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)

        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)

        return z_e, z_q

    def decode(self, z_q, quantify=False):
        if quantify:
            _, z_q, _, _ = self.vector_quantization(z_q)

        return self.decoder(z_q)


class VQVAE_2(nn.Module):
    def __init__(
            self,
            in_chan=3,
            h_dim=128,
            res_h_dim=32,
            n_res_layers=2,
            n_embeddings=512,
            embedding_dim=64,
            beta=0.25,
            save_img_embedding_map=False,
            batch_norm=False
    ):
        super(VQVAE_2, self).__init__()
        self.in_chan = in_chan
        self.embedding_dim = embedding_dim
        # encode image into continuous latent space
        self.encoder = RasterEncoder_2(in_chan, h_dim, n_res_layers, res_h_dim, batch_norm=batch_norm)
        self.pre_quantization_conv = nn.Conv2d(h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = RasterDecoder_2(embedding_dim, h_dim, n_res_layers, res_h_dim, in_chan, batch_norm=batch_norm)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def reconstruct(self, x):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        z_q, _, _ = self.vector_quantization.strait_through(z_e)
        x_hat = self.decoder(z_q)

        return x_hat, z_e, z_q

    def forward(self, x, verbose=False):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)

        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity, z_q

    def encode(self, x):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)

        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)

        return z_e, z_q

    def decode(self, z_q, quantify=False):
        if quantify:
            _, z_q, _, _ = self.vector_quantization(z_q)

        return self.decoder(z_q)


class VQVAE_3(nn.Module):
    def __init__(
            self,
            in_chan=3,
            h_dim=128,
            res_h_dim=32,
            n_res_layers=2,
            n_embeddings=512,
            embedding_dim=64,
            beta=0.25,
            save_img_embedding_map=False,
            batch_norm=False
    ):
        super(VQVAE_3, self).__init__()
        self.in_chan = in_chan
        self.embedding_dim = embedding_dim
        # encode image into continuous latent space
        self.encoder = RasterEncoder_3(in_chan, h_dim, n_res_layers, res_h_dim, batch_norm=batch_norm)
        self.pre_quantization_conv = nn.Conv2d(h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = RasterDecoder_3(embedding_dim, h_dim, n_res_layers, res_h_dim, in_chan, batch_norm=batch_norm)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def reconstruct(self, x):
        z_e, indices = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        z_q, _, _ = self.vector_quantization.strait_through(z_e)
        x_hat = self.decoder(z_q, indices)

        return x_hat, z_e, z_q

    def forward(self, x, verbose=False):
        z_e, indices = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)

        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q, indices)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity, z_q

    def encode(self, x):
        z_e, indices = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)

        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)

        return z_e, z_q

    def decode(self, z_q, indices, quantify=False):
        if quantify:
            _, z_q, _, _ = self.vector_quantization(z_q)

        return self.decoder(z_q, indices)


class VQVAE_4(nn.Module):
    def __init__(
            self,
            in_chan=3,
            h_dim=128,
            res_h_dim=32,
            n_res_layers=2,
            n_embeddings=512,
            embedding_dim=64,
            beta=0.25,
            save_img_embedding_map=False,
            batch_norm=False
    ):
        super(VQVAE_4, self).__init__()
        self.in_chan = in_chan
        self.embedding_dim = embedding_dim
        # encode image into continuous latent space
        self.encoder = RasterEncoder_4(in_chan, h_dim, n_res_layers, res_h_dim, batch_norm=batch_norm)
        self.pre_quantization_conv = nn.Conv2d(h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = RasterDecoder_4(embedding_dim, h_dim, n_res_layers, res_h_dim, in_chan, batch_norm=batch_norm)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def reconstruct(self, x):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        z_q, _, _ = self.vector_quantization.strait_through(z_e)
        x_hat = self.decoder(z_q)

        return x_hat, z_e, z_q

    def forward(self, x, verbose=False):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)

        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity, z_q

    def encode(self, x):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)

        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)

        return z_e, z_q

    def decode(self, z_q, quantify=False):
        if quantify:
            _, z_q, _, _ = self.vector_quantization(z_q)

        return self.decoder(z_q)


class VQVAE_5(nn.Module):
    def __init__(
            self,
            in_chan=3,
            h_dim=128,
            res_h_dim=32,
            n_res_layers=2,
            n_embeddings=512,
            embedding_dim=64,
            beta=0.25,
            save_img_embedding_map=False,
            batch_norm=False
    ):
        super(VQVAE_5, self).__init__()
        self.in_chan = in_chan
        self.embedding_dim = embedding_dim
        # encode image into continuous latent space
        self.encoder = RasterEncoder_4(in_chan, h_dim, n_res_layers, res_h_dim, batch_norm=batch_norm)
        self.pre_quantization_conv = nn.Conv2d(h_dim, embedding_dim, kernel_size=1, stride=1)
        self.vector_sequeneze_transfromer = nn.Transformer.encoder()
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = RasterDecoder_4(embedding_dim, h_dim, n_res_layers, res_h_dim, in_chan, batch_norm=batch_norm)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def reconstruct(self, x):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        z_q, _, _ = self.vector_quantization.strait_through(z_e)
        x_hat = self.decoder(z_q)

        return x_hat, z_e, z_q

    def forward(self, x, verbose=False):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)

        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity, z_q

    def encode(self, x):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)

        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)

        return z_e, z_q

    def decode(self, z_q, quantify=False):
        if quantify:
            _, z_q, _, _ = self.vector_quantization(z_q)

        return self.decoder(z_q)


class GDVQVAE(nn.Module):
    def __init__(
            self,
            in_chan=3,
            h_dim=128,
            res_h_dim=32,
            n_res_layers=2,
            n_embeddings=512,
            embedding_dim=64,
            beta=0.25,
            save_img_embedding_map=False,
            batch_norm=False,
            scale_lvl=2,
            kernel=4,
            scale_kernel=4,
            padding_mode='zeros'
    ):
        super(GDVQVAE, self).__init__()
        self.in_chan = in_chan
        self.embedding_dim = embedding_dim
        self.scale_lvl = scale_lvl
        self.scale_kernel = scale_kernel
        # encode image into continuous latent space
        self.encoder = GeneralRasterEncoder(
            in_dim=in_chan,
            h_dim=h_dim,
            n_res_layers=n_res_layers,
            res_h_dim=res_h_dim,
            down_scale=scale_lvl,
            start_kernel=kernel,
            down_kernel=scale_kernel,
            batch_norm=batch_norm,
            padding_mode=padding_mode
        )
        self.pre_quantization_conv = nn.Conv2d(h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = GeneralRasterDecoder(
            in_dim=embedding_dim,
            h_dim=h_dim,
            n_res_layers=n_res_layers,
            res_h_dim=res_h_dim,
            out_chan=in_chan,
            up_scale=scale_lvl,
            end_kernel=kernel,
            up_kernel=scale_kernel,
            batch_norm=batch_norm,
            padding_mode='zeros'
        )

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def reconstruct(self, x):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        z_q, _, _ = self.vector_quantization.strait_through(z_e)
        x_hat = self.decoder(z_q)

        return x_hat, z_e, z_q

    def forward(self, x, verbose=False):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)

        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity, z_q

    def encode(self, x):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)

        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)

        return z_e, z_q

    def decode(self, z_q, quantify=False):
        if quantify:
            _, z_q, _, _ = self.vector_quantization(z_q)

        return self.decoder(z_q)


class VQVAE_SLP(nn.Module):
    def __init__(
            self,
            in_chan=3,
            h_dim=128,
            res_h_dim=32,
            n_res_layers=2,
            n_embeddings=512,
            embedding_dim=64,
            beta=0.25,
            save_img_embedding_map=False,
            batch_norm=False,
            col_kernel=64
    ):
        super(VQVAE_SLP, self).__init__()
        self.in_chan = in_chan
        self.embedding_dim = embedding_dim
        self.col_kernel = col_kernel
        # encode image into continuous latent space
        self.encoder = Encoder(in_chan, h_dim, n_res_layers, res_h_dim, col_kernel=col_kernel, batch_norm=batch_norm)
        self.pre_quantization_conv = nn.Conv2d(h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim, in_chan, exp_kernel=col_kernel, batch_norm=batch_norm)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def reconstruct(self, x):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        z_q, _, _ = self.vector_quantization.strait_through(z_e)
        x_hat = self.decoder(z_q)

        return x_hat, z_e, z_q

    def forward(self, x, verbose=False):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)

        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity, z_q

    def encode(self, x):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)

        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)

        return z_e, z_q

    def decode(self, z_q, quantify=False):
        if quantify:
            _, z_q, _, _ = self.vector_quantization(z_q)

        return self.decoder(z_q)

