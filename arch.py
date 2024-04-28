from functools import partial
import math
import numpy as np

from einops.layers.torch import Rearrange, Reduce
from einops import rearrange
from regex import R
from torchvision.ops import Permute
import torch.nn.functional as F
from typing import Callable, Optional, List
import torch
from torch import nn, Tensor
from torch.nn import functional as F


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
    def __init__(self, dim, out_dim, heads = 8, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, out_dim, 1)

    def forward(self, x, t=None):
        b, c, z = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) z -> b h c (z)', h = self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c z -> b (h c) z', h = self.heads, z = z)
        return self.to_out(out)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x,):
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
        self,
        dim,
        layer_scale: float = 1e-4,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        kernel_size = 9
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.block = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, 
                      groups=dim, bias=True, padding_mode='replicate'),
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
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
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
        self.normalized_shape = (normalized_shape, )
    
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
    """ GRN (Global Response Normalization) layer
    """
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
        super().__init__(LayerNorm(in_ch, data_format='channels_first'), 
                         nn.Conv1d(in_ch, out_ch, 
                                                kernel_size=ks,
                                                stride=ks))
             
    
class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, mult=4, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim, padding_mode='reflect')
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, mult * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(mult * dim)
        self.pwconv2 = nn.Linear(mult * dim, dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 2, 1)

        return x + input


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )


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
    def __init__(self, dim, mult=2, block=Block,
                 scale_factor=2):
        super().__init__()
        self.first = block(dim)
        
        # if dim != dim_in:
        #     self.lin = nn.Conv1d(dim_in, dim, 1)
        # else: 
        #     self.lin = nn.Identity()
        
        self.down = DownSample(dim, dim*mult, ks=scale_factor)
        self.middle = Block(dim*mult)
        self.up = nn.Sequential(nn.Conv1d(dim*mult, dim*scale_factor, 1, groups=dim),
                                PixelShuffle1D(scale_factor))
        
        self.final = nn.Sequential(Block(dim*2, mult=2), nn.Conv1d(dim*2, dim, 1, groups=dim), )
        
    def forward(self, x_in):
        x1 = self.first(x_in)
        x = self.down(x1)
        x = self.middle(x)
        x = self.up(x)
        x = torch.cat([x, x1], dim=1)
        x = self.final(x)
        return x + x_in
    
        




class Net(nn.Module):
    def __init__(self, num_in_2d,  num_3d_in, num_2d_out, num_3d_out,
                 num_vert=60, dim=256, 
                 depth=4, block = UNetDepthOne,
                 frac_idxs = None):
        super().__init__()
        self.num_2d_in = num_in_2d
        self.num_3d_in = num_3d_in
        self.num_2d_out = num_2d_out
        self.num_3d_out = num_3d_out
        self.num_vert = num_vert
        self.frac_idxs = frac_idxs
                
        self.layer_2d_3d = nn.Sequential(nn.Conv1d(num_in_2d, num_in_2d*num_vert, kernel_size=1, groups=num_in_2d),
                                         Rearrange('b (c z) x -> b c (z x)', c=num_in_2d, z=num_vert, x=1))
        
        self.layer_3d = nn.Sequential(Rearrange('b (c z) -> b c z', c=num_3d_in, z=num_vert),
                                      nn.Conv1d(num_3d_in, dim - num_in_2d, groups=num_3d_in, kernel_size=1))
        
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
        
                
        layers = []
        for n in range(depth):
            layers.append(block(dim))
        # for n in range(num_blocks):
        #     layers.append(block(mid_ch))
        #     # if include_atten:
        #     #     atten = nn.Sequential(self.emb, LinearAttention(mid_ch+8, mid_ch))
        #     #     layers.append(Residual(PreNorm(mid_ch, atten)))
        #     layers.append(block(mid_ch))

            
        self.blocks = nn.Sequential(*layers)
        
        self.out_3d = nn.Sequential(Block(dim), 
                                    nn.Conv1d(dim, num_3d_out, 1), 
                                    Rearrange('b c z -> b (c z)'),
                                    nn.Linear(num_3d_out*num_vert, num_3d_out*num_vert))
        
        self.out_2d = nn.Sequential(Block(dim), 
                                    nn.Conv1d(dim, num_2d_out, 1),
                                    Reduce('b c z -> b c', 'mean'))
        
    def forward(self, x):
        split_idx = self.num_3d_in * self.num_vert

        x_3d, x_2d = x[:, :split_idx], x[:, split_idx:]
        x_out = torch.cat([self.layer_2d_3d(x_2d[..., None]), self.layer_3d(x_3d)], dim=1)
        x_out =  self.blocks(x_out)
        
        out_3d = self.out_3d(x_out)
        if self.frac_idxs is not None:
            s,e = self.frac_idxs
            
            out_frac = out_3d[:, s:e]*x_3d[:, s:e]
            out_3d[:, s:e] = out_frac
        
        out_2d = self.out_2d(x_out)
        
        return torch.cat([out_3d, out_2d], dim=1)

class sparseKernel1d(nn.Module):
    def __init__(self,
                 k, alpha, c=1,
                 nl=1,
                 initializer=None,
                 **kwargs):
        super(sparseKernel1d, self).__init__()

        self.k = k
        self.conv = self.convBlock(c*k, c*k)
        self.Lo = nn.Linear(c*k, c*k)

    def forward(self, x):
        B, N, c, ich = x.shape  # (B, N, c, k)
        x = x.view(B, N, -1)

        x_n = x.permute(0, 2, 1)
        x_n = self.conv(x_n)
        x_n = x_n.permute(0, 2, 1)

        x = self.Lo(x_n)
        x = x.view(B, N, c, ich)
        return x

    def convBlock(self, ich, och):
        net = nn.Sequential(
            nn.Conv1d(ich, och, 3, 1, 1),
            nn.GELU(),
            #nn.Conv1d(och, och, 3, 1, 1),
            # nn.GroupNorm(8, och,),
            # nn.GELU(),
        )
        return net
