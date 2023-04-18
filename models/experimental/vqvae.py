
import torch
import torch.nn as nn
import numpy as np
from models.vqvae.rasterencoder import GeneralRasterEncoder, RasterEncoder, RasterEncoder_2, RasterEncoder_3, RasterEncoder_4, Encoder
from models.vqvae.quantizer import VectorQuantizer
from models.vqvae.rasterdecoder import GeneralRasterDecoder, RasterDecoder, RasterDecoder_2, RasterDecoder_3, RasterDecoder_4, Decoder
from models.experimental.util import PositionalEncoding

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
            batch_norm=False,
            d_model=256,
            nhead=8,
            dropout=0.1,
            max_sequence_length=128,
    ):
        super(VQVAE_5, self).__init__()
        self.in_chan = in_chan
        self.embedding_dim = embedding_dim
        # encode image into continuous latent space
        self.encoder = RasterEncoder_2(in_chan, h_dim, n_res_layers, res_h_dim, batch_norm=batch_norm)
        self.pre_quantization_conv = nn.Conv2d(h_dim, embedding_dim, kernel_size=1, stride=1)
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_sequence_length)
        self.vector_sequeneze_transfromer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.sequeneze_vector_transfromer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
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
        z_q_shape = z_q.shape
        z_q = z_q.view(z_q.shape[0], self.embedding_dim, -1)
        z_q = self.positional_encoding(z_q)
        z_q = self.vector_sequeneze_transfromer(z_q)
        z_q = self.vector_sequeneze_transfromer(z_q)

        z_qe = self.sequeneze_vector_transfromer(z_q, z_q)
        z_qe = self.sequeneze_vector_transfromer(z_qe, z_q)
        z_qe = z_qe.view(z_q_shape)

        x_hat = self.decoder(z_qe)

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

