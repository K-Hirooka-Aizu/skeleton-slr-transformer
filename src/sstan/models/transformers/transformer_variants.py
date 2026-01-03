import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import Callable,Optional

# My library
from .modules.pos_encode import SinusoidalPositionalEncoding
from .modules.attention import (
    MultiHeadSelfAttention,
    RelativePositionalEncodeMultiHeadSelfAttention,
)

class FeedForwardNetwork(nn.Module):
    def __init__(self,in_channels:int, expand_ratio:float=1.0, dropout_ratio:float=0.0, bias:bool=False):
        super().__init__()

        mid_channels = int(in_channels*expand_ratio)
        self.linear_1 = nn.Linear(in_channels,mid_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_ratio) if dropout_ratio>0 else nn.Identity()
        self.linear_2 = nn.Linear(mid_channels,in_channels)

    def forward(self,x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x

class Block(nn.Module):
    def __init__(self, input_dims:int, n_heads:int, head_dims:int, seq_len:int, ffn_expand_ratio:float, ffn_dropout_ratio:float):
        super().__init__()

        self.temporal_attention = RelativePositionalEncodeMultiHeadSelfAttention(input_dim=input_dims, head_dim=head_dims, n_heads=n_heads, seq_len=seq_len, bias=False)
        self.norm1 = nn.LayerNorm(input_dims)

        self.ffn = FeedForwardNetwork(in_channels=input_dims, expand_ratio=ffn_expand_ratio, dropout_ratio=ffn_dropout_ratio)
        self.norm2 = nn.LayerNorm(input_dims)

    def forward(self, x):
        x = x + self.temporal_attention(x)
        x = self.norm1(x)

        x = x + self.ffn(x)
        x = self.norm2(x)

        return x
    
class TransformerVariant1(nn.Module):
    def __init__(self, in_channels:int, num_classes:int, seq_len:int, n_joints:int, embedding_dims:int,  n_blocks:int, n_heads:int, head_dims:int, ffn_expand_ratio:float=4.0, ffn_dropout_ratio:float=0.3, **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.embedding_dims = embedding_dims
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.n_joints = n_joints
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.head_dims = head_dims
        self.ffn_expand_ratio = ffn_expand_ratio
        self.ffn_dropout_ratio = ffn_dropout_ratio

        self.embedding = nn.Linear(in_channels, embedding_dims)
        self.blocks = nn.ModuleList(
            [
                Block(
                    input_dims=embedding_dims*self.n_joints,
                    n_heads=n_heads,
                    head_dims=head_dims,
                    seq_len=seq_len,
                    ffn_expand_ratio=ffn_expand_ratio,
                    ffn_dropout_ratio=ffn_dropout_ratio
                ) for _ in range(self.n_blocks)
            ]
        )

        self.fn = nn.Linear(embedding_dims, num_classes)

    def forward(self, x:torch.Tensor):

        B,C,T,V,M = x.size()
        x = x.transpose(1,-1).contiguous().view(-1,T,V,C)
        x = self.embedding(x) # (B*M, T, V, embed)

        x = x.view(-1, T, V * self.embedding_dims) # (B*M, T, V*embed)

        for block in self.blocks:
            x = block(x)

        x = x.view(B, M, T, V*self.embedding_dims).transpose(1,-1).contiguous()
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.fn(x)
        return x

        


