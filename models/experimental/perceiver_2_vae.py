from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
import models.utils as utils


# helpers

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


def fourier_encode(x, max_freq, num_bands=4, base=2):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.logspace(1., log(max_freq / 2) / log(base), num_bands, base=base, device=device, dtype=dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1)
    return x


# helper classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0., out_dim=None):
        super().__init__()
        out_dim = default(out_dim, dim)

        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., query_dim_out=None):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        query_dim_out = default(query_dim_out, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim_out),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None, get_attention_map=False):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        if get_attention_map:
            return self.to_out(out), attn
        return self.to_out(out)


class CollapsingAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., query_dim_out=None):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        query_dim_out = default(query_dim_out, query_dim)

        self.query_dim = query_dim
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim_out),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None, get_attention_map=False):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b j', q, k) * self.scale / self.query_dim

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b j, b j d -> b d', attn, v)
        out = rearrange(out, '(b h) d -> b (h d)', h=h)
        if get_attention_map:
            return self.to_out(out), attn
        return self.to_out(out)


class ExplodingAttention(nn.Module):
    def __init__(self, query_dim, explosion_dim=None, context_dim=None, heads=8, dim_head=64, dropout=0., query_dim_out=None):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        query_dim_out = default(query_dim_out, query_dim)

        self.query_dim = query_dim
        self.explosion_dim = explosion_dim
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim_out),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None, get_attention_map=False):
        h = self.heads
        # maybe repeat at different place?
        x = repeat(x, 'b d -> b n d', n=self.explosion_dim)
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        if get_attention_map:
            return self.to_out(out), attn
        return self.to_out(out)


# base class
class PerceiverVLT(nn.Module):
    def __init__(
            self,
            *,
            num_freq_bands,
            depth,
            max_freq,
            freq_base=2,
            input_channels=3,
            input_axis=2,
            num_latents=512,
            latent_dim=512,
            cross_heads=1,
            latent_heads=8,
            cross_dim_head=64,
            latent_dim_head=64,
            attn_dropout=0.,
            ff_dropout=0.,
            weight_tie_layers=False
    ):
        super().__init__()
        self.input_axis = input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.freq_base = freq_base

        input_dim = input_axis * ((num_freq_bands * 2) + 1) + input_channels

        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        # self.latents_to_latent = PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout, out_dim=1))
        self.latents_to_latent = PreNorm(latent_dim, CollapsingAttention(latent_dim, heads=latent_heads, dim_head=latent_dim_head, dropout=attn_dropout))
        self.last_mu = PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))
        self.last_var = PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))

        get_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, input_dim, heads=cross_heads, dim_head=cross_dim_head, dropout=attn_dropout), context_dim=input_dim)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head, dropout=attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))
        if type(weight_tie_layers) is bool:
            weight_tie_layers = [weight_tie_layers] * depth
        assert len(weight_tie_layers) == depth, "weight_tie_layers must either be bool or an enumerable with len equal to depth"
        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers[i]
            cache_args = {'_cache': should_cache}

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

    def forward(self, data, mask=None):
        b, *axis, _, device = *data.shape, data.device
        assert len(axis) == self.input_axis, f'input data must have the right number of axis! Data: {len(axis)} Axis: {self.input_axis}'

        # calculate fourier encoded positions in the range of [-1, 1], for all axis

        axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device), axis))

        pos = torch.stack(torch.meshgrid(*axis_pos), dim=-1)
        enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands, base=self.freq_base)
        enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
        enc_pos = repeat(enc_pos, '... -> b ...', b=b)

        # concat to channels of data and flatten axis

        data = torch.cat((data, enc_pos), dim=-1)
        data = rearrange(data, 'b ... d -> b (...) d')

        x = repeat(self.latents, 'n d -> b n d', b=b)

        for cross_attn, cross_ff, latent_attn, latent_ff in self.layers:
            x = cross_attn(x, context=data, mask=mask) + x
            x = cross_ff(x) + x
            x = latent_attn(x) + x
            x = latent_ff(x) + x

        x = self.latents_to_latent(x) + x.mean(dim=-2)
        mu, logvar = self.last_mu(x) + x, self.last_var(x) + x
        z = utils.sample(mu, logvar)
        if self.training:
            return z, mu, logvar
        return z

    def forward_with_attention_maps(self, data, mask=None):
        b, *axis, _, device = *data.shape, data.device
        assert len(axis) == self.input_axis, 'input data must have the right number of axis'

        # calculate fourier encoded positions in the range of [-1, 1], for all axis
        axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device), axis))
        pos = torch.stack(torch.meshgrid(*axis_pos), dim=-1)
        enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands, base=self.freq_base)
        enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
        enc_pos = repeat(enc_pos, '... -> b ...', b=b)

        # concat to channels of data and flatten axis
        data = torch.cat((data, enc_pos), dim=-1)
        data = rearrange(data, 'b ... d -> b (...) d')

        x = repeat(self.latents, 'n d -> b n d', b=b)
        attn_maps = []
        for cross_attn, cross_ff, latent_attn, latent_ff in self.layers:
            y, cur_attn_map = cross_attn(x, context=data, mask=mask, get_attention_map=True)
            x = x + y
            x = cross_ff(x) + x
            x = latent_attn(x) + x
            x = latent_ff(x) + x
            attn_maps += [cur_attn_map]

        x, cur_attn_map = self.latents_to_latent(x, get_attention_map=True)
        attn_maps += [cur_attn_map]

        x = self.latents_to_latent(x) + x.mean(dim=-2)
        mu, logvar = self.last_mu(x) + x, self.last_var(x) + x
        z = utils.sample(mu, logvar)
        if self.training:
            return z, mu, logvar, attn_maps
        return z, attn_maps
