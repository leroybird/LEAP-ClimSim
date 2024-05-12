from collections import namedtuple
from rotary_embedding_torch import RotaryEmbedding
from functools import partial
import math
import re
import numpy as np

from einops.layers.torch import Rearrange, Reduce
from einops import rearrange
import pandas as pd
from regex import R
from torchvision.ops import Permute
import torch.nn.functional as F
from typing import Callable, Optional, List, final
import torch
from torch import Block, nn, Tensor
from torch.nn import functional as F

from mwt import MWT1d


class SinusoidalPosEmb(nn.Module):
    """
    Based on transformer-like embedding from 'Attention is all you need'
    Note: 10,000 corresponds to the maximum sequence length
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Embedder(nn.Module):
    def __init__(self, num_vert, dim=16):
        super().__init__()
        self.emb = SinusoidalPosEmb(dim)
        self.num_vert = num_vert

    def forward(self, x):
        x_b = x.shape[0]
        emb = self.emb(torch.arange(0, self.num_vert)).to(x.device).T
        emb = emb[None, :, :].repeat(x_b, 1, 1)
        return torch.cat([x, emb], dim=1)


class LinearAttention(nn.Module):
    def __init__(self, dim, out_dim, heads=8, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, out_dim, 1)

    def forward(self, x, t=None):
        b, c, z = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) z -> b h c (z)", h=self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim=-1)
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c z -> b (h c) z", h=self.heads, z=z)
        return self.to_out(out)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(
        self,
        x,
    ):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class CNBlock1d(nn.Module):
    def __init__(
        self, dim, layer_scale: float = 1e-4, norm_layer: Optional[Callable[..., nn.Module]] = None, kernel_size=9
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.block = nn.Sequential(
            nn.Conv1d(
                dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim, bias=True, padding_mode="replicate"
            ),
            Permute([0, 2, 1]),
            norm_layer(dim),
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            Permute([0, 2, 1]),
        )
        self.layer_scale = nn.Parameter(torch.ones(1, dim, 1) * layer_scale)

    def forward(self, input: Tensor) -> Tensor:
        result = self.layer_scale * self.block(input)
        return result + input


class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class DownSample(nn.Sequential):
    def __init__(self, in_ch, out_ch=None, ks=4):
        if out_ch is None:
            out_ch = in_ch
        super().__init__(LayerNorm(in_ch, data_format="channels_first"), nn.Conv1d(in_ch, out_ch, kernel_size=ks, stride=ks))


class ConvNextBlock2(nn.Module):
    """ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim_in, dim=None, mult=4, drop_path=0.0, ks=7):
        if dim is None:
            dim = dim_in

        super().__init__()
        self.lin = nn.Conv1d(dim_in, dim, 1) if dim_in != dim else nn.Identity()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=ks, padding=ks // 2, groups=dim, padding_mode="reflect")
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, mult * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(mult * dim)
        self.pwconv2 = nn.Linear(mult * dim, dim)
        self.identity = nn.Identity() if dim == dim_in else nn.Conv1d(dim_in, dim, 1)

    def forward(self, x):
        inp = x
        x = self.dwconv(self.lin(x))
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 1)

        return x + self.identity(inp)


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForwardFlip(dim, expansion_factor=4, dropout=0.0, dense=nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(dense(dim, inner_dim), nn.GELU(), nn.Dropout(dropout), dense(inner_dim, dim), nn.Dropout(dropout))


class PixelShuffle1D(nn.Module):
    """
    1D pixel shuffler. https://arxiv.org/pdf/1609.05158.pdf
    Upscales sample length, downscales channel length
    "short" is input, "long" is output
    """

    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width

        x = x.contiguous().view([batch_size, self.upscale_factor, long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)

        return x


class UNetDepthOne(nn.Module):
    def __init__(self, dim, mult=2, block=ConvNextBlock2, scale_factor=2):
        super().__init__()
        self.first = block(dim)

        # if dim != dim_in:
        #     self.lin = nn.Conv1d(dim_in, dim, 1)
        # else:
        #     self.lin = nn.Identity()

        self.down = DownSample(dim, dim * mult, ks=scale_factor)
        self.middle = ConvNextBlock2(dim * mult)
        self.up = nn.Sequential(nn.Conv1d(dim * mult, dim * scale_factor, 1, groups=dim), PixelShuffle1D(scale_factor))

        self.final = nn.Sequential(
            ConvNextBlock2(dim * 2, mult=2),
            nn.Conv1d(dim * 2, dim, 1, groups=dim),
        )

    def forward(self, x_in):
        x1 = self.first(x_in)
        x = self.down(x1)
        x = self.middle(x)
        x = self.up(x)
        x = torch.cat([x, x1], dim=1)
        x = self.final(x)
        return x + x_in


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


FlashAttentionConfig = namedtuple("FlashAttentionConfig", ["enable_flash", "enable_math", "enable_mem_efficient"])


class Attend(nn.Module):
    def __init__(self, dropout=0.0, flash=False, scale=None):
        super().__init__()
        self.scale = scale
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = FlashAttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))

        # if device_properties.major == 8 and device_properties.minor == 0:
        #     print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
        self.cuda_config = FlashAttentionConfig(True, False, False)
        # else:
        #     print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
        #     self.cuda_config = FlashAttentionConfig(False, True, True)

    def flash_attn(self, q, k, v):
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device
        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, softmax_scale
        # print(q.shape, k.shape, v.shape)

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)

        return out

    def forward(self, q, k, v):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        return self.flash_attn(q, k, v)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, rotary_embed=None, flash=True):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed

        self.attend = Attend(flash=flash, dropout=dropout)

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)

        self.to_gates = nn.Linear(dim, heads)

        self.to_out = nn.Sequential(nn.Linear(dim_inner, dim, bias=False), nn.Dropout(dropout))

    def forward(self, x):
        x = self.norm(x)

        q, k, v = rearrange(self.to_qkv(x), "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads)

        if self.rotary_embed is not None:
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        out = self.attend(q, k, v)

        gates = self.to_gates(x)
        out = out * rearrange(gates, "b n h -> b h n 1").sigmoid()

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ff_mult=4,
        norm_output=True,
        rotary_embed=None,
        flash_attn=True
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            dropout=attn_dropout,
                            rotary_embed=rotary_embed,
                            flash=flash_attn,
                        ),
                        FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class UnetConvnext(nn.Module):
    def __init__(self, num_in, dim, dim_mults=(1, 2, 4), residual=False, block=partial(ConvNextBlock2, ks=5)):
        super().__init__()
        self.residual = residual

        dims = [num_in, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        pos_embs = []
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            heads = dim_out // 64
            pos_emb = RotaryEmbedding(dim=64)
            pos_embs.append(pos_emb)

            self.downs.append(
                nn.ModuleList(
                    [
                        nn.Sequential(
                            block(dim_in, dim_out),
                            Rearrange("b c z -> b z c"),
                            Transformer(dim=dim_out, depth=1, dim_head=64, heads=heads, rotary_embed=pos_emb),
                            Rearrange("b z c -> b c z"),
                            block(dim_out, dim_out),
                        ),
                        DownSample(dim_out, ks=2) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block(mid_dim, mid_dim)
        self.mid_attn = nn.Sequential(
            Rearrange("b c z -> b z c"),
            Transformer(dim=mid_dim, depth=1, dim_head=64, heads=heads, rotary_embed=pos_embs[-1]),
            Rearrange("b z c -> b c z"),
        )

        self.mid_block2 = block(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            heads = dim_in // 64
            pos_emb = pos_embs.pop()

            self.ups.append(
                nn.ModuleList(
                    [
                        nn.Sequential(
                            block(dim_out * 2, dim_in),
                            Rearrange("b c z -> b z c"),
                            Transformer(dim=dim_in, depth=1, dim_head=64, heads=heads, rotary_embed=pos_emb),
                            Rearrange("b z c -> b c z"),
                            block(dim_in, dim_in),
                        ),
                        nn.Upsample(scale_factor=2, mode="linear", align_corners=True) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = block(dim)

    def forward(self, x, time=None):
        orig_x = x
        h = []

        for convnext, downsample in self.downs:
            x = convnext(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        for convnext, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = convnext(x)
            x = upsample(x)

        if self.residual:
            return self.final_conv(x) + orig_x

        out = self.final_conv(x)

        return out


class Net(nn.Module):
    def __init__(
        self,
        num_in_2d,
        num_3d_in,
        num_2d_out,
        num_3d_out,
        num_3d_start=6,
        num_vert=60,
        dim=512,
        depth=8,
        block=UNetDepthOne,
        num_emb=384,
        emb_ch=32,
        use_emb: bool = False,
        frac_idxs=None,
    ):
        super().__init__()
        self.num_2d_in = num_in_2d

        if use_emb:
            self.embedding = nn.Embedding(num_emb, emb_ch)
            num_in_2d += emb_ch
        else:
            self.embedding = None

        self.num_3d_in = num_3d_in
        self.num_2d_out = num_2d_out
        self.num_3d_out = num_3d_out
        self.num_vert = num_vert
        self.frac_idxs = frac_idxs
        self.num_3d_start = num_3d_start
        self.layer_2d_3d = nn.Sequential(
            nn.Conv1d(num_in_2d, num_in_2d * num_vert, kernel_size=1, groups=num_in_2d),
            Rearrange("b (c z) x -> b c (z x)", c=num_in_2d, z=num_vert, x=1),
        )

        self.layer_3d = nn.Sequential(
            Rearrange("b (c z) -> b c z", c=num_3d_in, z=num_vert), nn.Conv1d(num_3d_in, dim - num_in_2d, kernel_size=1)
        )

        heads = dim // 64
        self.rotary_embed = RotaryEmbedding(dim=dim//heads)

        # layers = []
        # for n in range(depth):
        #     layers.extend([Block(dim,),
        #                     Rearrange('b c z -> b z c'),
        #                    Transformer(dim=dim, depth=1, dim_head=dim//heads,
        #                                         heads=heads, rotary_embed=self.rotary_embed),
        #                     Rearrange('b z c -> b c z')])
        # self.blocks = nn.Sequential(*layers)
        
        self.blocks = nn.Sequential(
            block(dim),
            Rearrange("b c z -> b z c"),
            Transformer(dim=dim, depth=depth, dim_head=dim // heads, heads=heads, rotary_embed=self.rotary_embed),
            Rearrange("b z c -> b c z"),
        )
        # self.blocks = UnetConvnext(num_in=dim, dim=dim, residual=False)

        # conv_down = partial(nn.Conv3d, kernel_size=[1, 3, 3], padding=(0, 0, 0), padding_mode='reflect')
        # layers = [conv_down(num_2d_3d + num_3d_in, mid_ch//2), nn.GELU(),
        #           conv_down(mid_ch//2, mid_ch), nn.GELU(),
        #           Rearrange('b c z y x -> b c (z y x)', y=1, x=1)]
        # self.emb = Embedder(num_vert, 8)
        # chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

        # layers = [Permute([0, 2, 1])]
        # for n in range(depth):
        #     layers.append(PreNormResidual(dim, FeedForward(num_vert, 4, dense=chan_first)))
        #     layers.append(PreNormResidual(dim, FeedForward(dim, 4, dense=chan_last)))
        # layers.append(nn.LayerNorm(dim))
        # layers.append(Permute([0, 2, 1]))

        # layers = []
        # for n in range(depth):
        #     layers.append(block(dim))
        # self.blocks = nn.Sequential(*layers)
        # self.blocks = MWT1d(dim, depth=depth)
        # for n in range(num_blocks):
        #     layers.append(block(mid_ch))
        #     # if include_atten:
        #     #     atten = nn.Sequential(self.emb, LinearAttention(mid_ch+8, mid_ch))
        #     #     layers.append(Residual(PreNorm(mid_ch, atten)))
        #     layers.append(block(mid_ch))

        self.final_mult = 16
        out_3d = num_3d_out * self.final_mult * num_vert

        # self.out_3d = nn.Sequential(Block(dim),
        #                             nn.Conv1d(dim, num_3d_out*self.final_mult, 1),
        #                             nn.GELU(),
        #                             LayerNorm(num_3d_out*self.final_mult, data_format='channels_first'),
        #                             Rearrange('b c (z h) -> b (c z) h', h=1),)

        # self.out_3d_2 = nn.Sequential(nn.Conv1d(out_3d, out_3d, 2, 2, groups=out_3d),
        #                                 LayerNorm(out_3d, data_format='channels_first'),
        #                                 nn.GELU(),
        #                                 nn.Conv1d(out_3d, num_3d_out*num_vert, 1, groups=num_3d_out),
        #                                 Rearrange('b c h -> b (c h)', h=1))

        self.out_3d = nn.Sequential(
            block(dim),
            nn.Conv1d(dim, num_3d_out, 1),
            Rearrange("b c z -> b (c z)"),
            nn.Linear(
                num_3d_out * num_vert,
                num_3d_out * num_vert,
            ),
        )

        self.out_2d = nn.Sequential(block(dim), nn.Conv1d(dim, num_2d_out, 1), 
                                    Reduce("b c z -> b c", "mean"))

    def forward(self, x):
        x, emb_idxs = x

        split_idx = self.num_3d_start * self.num_vert

        # Data contains 3d vars, 2d vars, then 3d vars again
        x_3d, x_2d = x[:, :split_idx], x[:, split_idx:]
        x_3d_pt2, x_2d = x_2d[:, self.num_2d_in :], x_2d[:, : self.num_2d_in]
        x_3d_pt2 = torch.cat([x_3d, x_3d_pt2], dim=1)

        if self.embedding is not None:
            x_emb = self.embedding(emb_idxs)
            x_2d = torch.cat([x_2d, x_emb], dim=1)

        x_out = torch.cat([self.layer_2d_3d(x_2d[..., None]), self.layer_3d(x_3d_pt2)], dim=1)
        x_out = self.blocks(x_out)

        out_3d = self.out_3d(x_out)

        # x_3d_rep = x_3d[:, :, None].repeat(1, self.final_mult, 1)
        # out_3d = self.out_3d_2(torch.cat([x_3d_rep, out_3d], dim=-1))

        if self.frac_idxs is not None:
            s, e = self.frac_idxs

            out_frac = out_3d[:, s:e] * x_3d[:, s:e]
            out_3d[:, s:e] = out_frac

        out_2d = self.out_2d(x_out)

        return torch.cat([out_3d, out_2d], dim=1)


class Transformer2(nn.Module):
    def __init__(self, dim, depth, heads=8, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNormResidual(dim, nn.MultiheadAttention(dim, heads, dropout=dropout)),
                        PreNormResidual(
                            dim,
                            FeedForward(
                                dim,
                            ),
                        ),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


class NetMLP(nn.Module):
    def __init__(self, num_in_2d, num_3d_in, num_2d_out, num_3d_out, num_3d_start=6, num_vert=60, depth=12, dim=512):
        super().__init__()
        self.num_2d_in = num_in_2d
        self.num_3d_in = num_3d_in
        self.num_2d_out = num_2d_out
        self.num_3d_out = num_3d_out
        self.num_vert = num_vert
        self.num_3d_start = num_3d_start

        layers = [nn.Linear(num_in_2d + num_vert * num_3d_in, dim)]
        layers += [PreNormResidual(dim, FeedForward(dim)) for _ in range(depth)]
        layers += [nn.Linear(dim, num_2d_out + num_3d_out * num_vert)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x, emb_idxs = x
        return self.layers(x)


class NetTr(nn.Module):
    def __init__(
        self,
        num_in_2d,
        num_3d_in,
        num_2d_out,
        num_3d_out,
        num_3d_start=6,
        num_vert=60,
        dim=512,
        depth=4,
        num_pos_emb=384,
        use_pos_emb: bool = False,
        frac_idxs=None,
    ):
        super().__init__()
        # self.num_2d_in = num_in_2d

        self.total_in = num_in_2d + num_3d_in * num_vert
        self.total_out = num_2d_out + num_3d_out * num_vert

        self.emb_in = nn.Linear(1, dim)

        self.emb_out = nn.Parameter(torch.randn(self.total_out, dim))
        self.vert_emb = nn.Embedding(num_vert + 1, dim)
        self.var_emb = nn.Embedding(num_3d_in + num_in_2d + num_2d_out + num_3d_out, dim)

        # Input idxs, 3d, 2d, 3d
        vert_indxs = torch.arange(0, num_vert, dtype=torch.long).repeat(num_3d_in)
        vert_indxs_2d = torch.Tensor([num_vert] * num_in_2d).long()
        vert_indxs = torch.cat([vert_indxs[0 : num_3d_start * num_vert], vert_indxs_2d, vert_indxs[num_3d_start * num_vert :]])

        # output idxs, 3d, 2d
        vert_indxs = torch.cat([vert_indxs, torch.arange(num_vert, dtype=torch.long).repeat(num_3d_out)])
        self.vert_indxs = torch.cat([vert_indxs, num_vert * torch.ones(num_2d_out, dtype=torch.long)])

        total_3d = num_3d_in + num_3d_out
        var_idxs = torch.Tensor([[n] * num_vert for n in range(total_3d)]).flatten().long()
        var_idxs_2d = torch.arange(total_3d, total_3d + num_in_2d + num_2d_out, step=1, dtype=torch.long)

        # 3d_in, 2d_in, 3d_in_2, 2d_out, 3d_out
        self.var_idxs = torch.cat(
            [
                var_idxs[0 : num_3d_start * num_vert],
                var_idxs_2d[:num_in_2d],
                var_idxs[num_3d_start * num_vert :],
                var_idxs_2d[num_in_2d:],
            ]
        )

        self.rotary_embed = RotaryEmbedding(dim=64)
        # Transformer
        self.transformer = Transformer(dim=dim, depth=depth, dim_head=64, heads=dim // 64, rotary_embed=self.rotary_embed)

        if use_pos_emb:
            self.pos_emb = nn.Embedding(num_pos_emb, dim)
        else:
            self.pos_emb = None

        self.out = nn.Linear(dim, 1)

    def check_emb_idxs(self, input_names, output_names):
        return pd.DataFrame(
            {
                "names": input_names + output_names,
                "var_idxs": self.var_idxs.cpu().numpy(),
                "vert_idxs": self.vert_indxs.cpu().numpy(),
            }
        )

    def forward(self, x):
        x, emb_idxs = x
        bs, num_in = x.shape
        # Treat x like a sequence of single channels values
        x_in = self.emb_in(x[:, :, None])
        x_out = self.emb_out[None, :, :].repeat(bs, 1, 1)
        x = torch.cat([x_in, x_out], dim=1)

        x = x + self.vert_emb(self.vert_indxs.to(x.device))[None, :, :]
        x = x + self.var_emb(self.var_idxs.to(x.device))[None, :, :]
        if self.pos_emb is not None:
            x = x + self.pos_emb(emb_idxs)

        x = self.transformer(x)
        x = x[:, num_in:, :]
        return self.out(x)[:, :, 0]
