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


class B2TPostNormSpatialTemporalTransformerBlock(nn.Module):
    def __init__(self,input_dim:int,head_dim:int,n_heads:int,seq_len:int,n_joints:int,ffn_expand_ratio:float=4.0,ffn_dropout_ratio:float=.3,norm_layer=nn.LayerNorm,stochastic_depth_rate=0.0,bias=False):
        super().__init__()
        
        self.multihead_self_attention1=MultiHeadSelfAttention(input_dim,head_dim,n_heads,bias=bias)
        self.norm_layer1 = norm_layer(input_dim) if norm_layer is nn.BatchNorm2d else norm_layer([seq_len, n_joints, input_dim])
        
        self.multihead_self_attention2=RelativePositionalEncodeMultiHeadSelfAttention(input_dim,head_dim,n_heads,seq_len,bias=bias)
        self.norm_layer2 = norm_layer(input_dim) if norm_layer is nn.BatchNorm2d else norm_layer([n_joints, seq_len, input_dim])
        
        self.feed_forward_network = FeedForwardNetwork(input_dim,ffn_expand_ratio,ffn_dropout_ratio,bias)
        self.norm_layer3 = norm_layer(input_dim) if norm_layer is nn.BatchNorm2d else norm_layer([seq_len, n_joints, input_dim])

        self.stochastic_depth = torchvision.ops.StochasticDepth(stochastic_depth_rate,mode="row") if stochastic_depth_rate > 0 else nn.Identity()

    def forward(self,input:torch.Tensor, mask=None)->torch.Tensor:
        """
            inputs:
                input : (batch, * , seq_len[temporal], vertex[spatial], feature)
            returns:
                x : the shape is same as input
        """
        input_shape = input.size()

        # spatial self-attention
        z, _ = self.multihead_self_attention1(input, mask)
        x = input + self.stochastic_depth(z)
        x = self.norm_layer1(x) if isinstance(self.norm_layer1,nn.LayerNorm) else self.norm_layer1(x.transpose(-1,1)).transpose(-1,1)

        # temporal self-attention
        x = x.transpose(-2,-3).contiguous()
        z, _ = self.multihead_self_attention2(x, mask.transpose(-1,-2)) if mask != None else self.multihead_self_attention2(x)
        x = x + self.stochastic_depth(z)
        x = self.norm_layer2(x) if isinstance(self.norm_layer2,nn.LayerNorm) else self.norm_layer2(x.transpose(-1,1)).transpose(-1,1)
        x = x.transpose(-2,-3).contiguous()
        
        # feed forward network
        x = x + self.stochastic_depth(self.feed_forward_network(x)) + input
        x = self.norm_layer3(x) if isinstance(self.norm_layer3,nn.LayerNorm) else self.norm_layer3(x.transpose(-1,1)).transpose(-1,1)

        return x

class PostNormSpatialTemporalTransformerBlock(nn.Module):
    def __init__(self,input_dim:int,head_dim:int,n_heads:int,seq_len:int,n_joints:int,ffn_expand_ratio:float=4.0,ffn_dropout_ratio:float=.3,norm_layer=nn.LayerNorm,stochastic_depth_rate=0.0,bias=False):
        super().__init__()
        
        self.multihead_self_attention1=MultiHeadSelfAttention(input_dim,head_dim,n_heads,bias=bias)
        self.norm_layer1 = norm_layer(input_dim) if norm_layer is nn.BatchNorm2d else norm_layer([seq_len, n_joints, input_dim])
        
        self.multihead_self_attention2=RelativePositionalEncodeMultiHeadSelfAttention(input_dim,head_dim,n_heads,seq_len,bias=bias)
        self.norm_layer2 = norm_layer(input_dim) if norm_layer is nn.BatchNorm2d else norm_layer([n_joints, seq_len, input_dim])
        
        self.feed_forward_network = FeedForwardNetwork(input_dim,ffn_expand_ratio,ffn_dropout_ratio,bias)
        self.norm_layer3 = norm_layer(input_dim) if norm_layer is nn.BatchNorm2d else norm_layer([seq_len, n_joints, input_dim])

        self.stochastic_depth = torchvision.ops.StochasticDepth(stochastic_depth_rate,mode="row") if stochastic_depth_rate > 0 else nn.Identity()

    def forward(self,input:torch.Tensor, mask=None)->torch.Tensor:
        """
            inputs:
                input : (batch, * , seq_len[temporal], vertex[spatial], feature)
            returns:
                x : the shape is same as input
        """
        input_shape = input.size()

        # spatial self-attention
        z, _ = self.multihead_self_attention1(input, mask)
        x = input + self.stochastic_depth(z)
        x = self.norm_layer1(x) if isinstance(self.norm_layer1,nn.LayerNorm) else self.norm_layer1(x.transpose(-1,1)).transpose(-1,1)

        # temporal self-attention
        x = x.transpose(-2,-3).contiguous()
        z, _ = self.multihead_self_attention2(x, mask.transpose(-1,-2)) if mask != None else self.multihead_self_attention2(x)
        x = x + self.stochastic_depth(z)
        x = self.norm_layer2(x) if isinstance(self.norm_layer2,nn.LayerNorm) else self.norm_layer2(x.transpose(-1,1)).transpose(-1,1)
        x = x.transpose(-2,-3).contiguous()
        
        # feed forward network
        x = x + self.stochastic_depth(self.feed_forward_network(x)) + input
        x = self.norm_layer3(x) if isinstance(self.norm_layer3,nn.LayerNorm) else self.norm_layer3(x.transpose(-1,1)).transpose(-1,1)

        return x



class SpatialTemporalTransformer(nn.Module):
    def __init__(
        self,
        in_channels:int,
        num_classes:int,
        seq_len:int,
        n_joints:int,
        embedding_dim:int,
        n_blocks:int,
        head_dim:int,
        n_heads:int,
        ffn_expand_ratio:float=4.0,
        ffn_dropout_ratio:float=0.25,
        max_stochastic_depth_rate:float=0.25,
        **kwargs
        ):
        super().__init__()

        self.in_channels=in_channels
        self.num_classes=num_classes
        self.seq_len=seq_len
        self.n_joints=n_joints
        self.embedding_dim=embedding_dim
        self.n_blocks=n_blocks
        self.head_dim=head_dim
        self.n_heads=n_heads
        self.ffn_expand_ratio = ffn_expand_ratio
        self.ffn_dropout_ratio = ffn_dropout_ratio
        self.max_stochastic_depth_rate=max_stochastic_depth_rate
        self.bias=True

        self.embedding = nn.Sequential(
            nn.Linear(in_channels,embedding_dim//2,bias=self.bias),
            nn.Tanh(),
            nn.Linear(embedding_dim//2,embedding_dim,bias=self.bias)
        )
        self.spatial_positional_encode = SinusoidalPositionalEncoding(embedding_dim,n_joints)

        self.blocks = nn.ModuleList([
            B2TPostNormSpatialTemporalTransformerBlock(
                input_dim = embedding_dim,
                head_dim = head_dim,
                n_heads = n_heads,
                seq_len=seq_len,
                n_joints=n_joints,
                ffn_expand_ratio=self.ffn_expand_ratio,
                ffn_dropout_ratio=self.ffn_dropout_ratio,
                norm_layer=nn.BatchNorm2d,
                # norm_layer=nn.LayerNorm,
                stochastic_depth_rate=stochastic_depth_rate,
                bias=self.bias,
                
            )
        for _,stochastic_depth_rate in zip(range(n_blocks),np.linspace(0.0,self.max_stochastic_depth_rate,n_blocks))])
        self.fc = nn.Linear(embedding_dim,num_classes,bias=self.bias)

    def forward(self,input:torch.Tensor) -> torch.Tensor:
        """
        forward process

        args:
        x : (batch, feature, time, vertex, body)
        """
        x = input
        x = self.extract_feature(x)
        x = torch.squeeze(F.adaptive_avg_pool3d(x,1))
        pred = self.fc(x)
        return pred

    def extract_feature(self,x:torch.Tensor, mask=None) -> torch.Tensor:
        """
        x : (batch, feature, time, vertex, body)
        """
        B,C,T,V,M = x.size()
        x = x.transpose(1,-1).contiguous().view(-1,T,V,C)

        x = self.embedding(x)

        x = self.spatial_positional_encode(x)

        for transformer_block in self.blocks:
            x = transformer_block(x,mask)

        x = x.view(B,M,T,V,self.embedding_dim).transpose(1,-1).contiguous()
        return x
        


class SpatialTemporalTransformerWithClassToken(nn.Module):
    def __init__(
        self,
        in_channels:int,
        num_classes:int,
        seq_len:int,
        n_joints:int,
        embedding_dim:int,
        n_blocks:int,
        head_dim:int,
        n_heads:int,
        ffn_expand_ratio:float=4.0,
        ffn_dropout_ratio:float=0.25,
        max_stochastic_depth_rate:float=0.25,
        **kwargs):
        super().__init__()

        self.in_channels=in_channels
        self.num_classes=num_classes
        self.seq_len=seq_len + 1
        self.n_joints=n_joints
        self.embedding_dim=embedding_dim
        self.n_blocks=n_blocks
        self.head_dim=head_dim
        self.n_heads=n_heads
        self.ffn_expand_ratio = ffn_expand_ratio
        self.ffn_dropout_ratio = ffn_dropout_ratio
        self.max_stochastic_depth_rate=max_stochastic_depth_rate
        self.bias=True

        self.cls_token = nn.Parameter(torch.zeros(1,1,self.embedding_dim))

        self.embedding = nn.Sequential(
            # nn.Linear(in_channels,self.embedding_dim//2,bias=self.bias),
            # nn.Tanh(),
            # nn.Linear(self.embedding_dim//2,self.embedding_dim,bias=self.bias)
            nn.Linear(in_channels,self.embedding_dim,bias=self.bias)
        )
        self.spatial_positional_encode = SinusoidalPositionalEncoding(embedding_dim,n_joints)

        self.blocks = nn.ModuleList([
            
            B2TPostNormSpatialTemporalTransformerBlock(
                input_dim = embedding_dim,
                head_dim = head_dim,
                n_heads = n_heads,
                seq_len=self.seq_len,
                n_joints=self.n_joints,
                ffn_expand_ratio=self.ffn_expand_ratio,
                ffn_dropout_ratio=self.ffn_dropout_ratio,
                norm_layer=nn.BatchNorm2d,
                # norm_layer=nn.LayerNorm,
                stochastic_depth_rate=stochastic_depth_rate,
                bias=self.bias,
            )
        for _,stochastic_depth_rate in zip(range(n_blocks),np.linspace(0.0,self.max_stochastic_depth_rate,n_blocks))])
        
        self.fc = nn.Linear(self.embedding_dim,num_classes,bias=self.bias)


    def forward(self,input:torch.Tensor) -> torch.Tensor:
        """
        x : (batch, feature, time, vertex, body)
        """
        x = input
        x = self.extract_feature(x)
        pred = self.fc(x)
        return pred

    def extract_feature(self,x:torch.Tensor, mask=None) -> torch.Tensor:
        """
        x : (batch, feature, time, vertex, body)
        """
        B,C,T,V,M = x.size()
        x = x.transpose(1,-1).contiguous().view(-1,T,V,C)

        x = self.embedding(x)
        cls_token = self.cls_token.expand(B*M,-1,V,-1)
        x = torch.cat([cls_token, x], dim=1)
        
        x = self.spatial_positional_encode(x)

        for transformer_block in self.blocks:
            x = transformer_block(x, mask)

        x = x[:,0].contiguous().view(B,V,-1).mean(dim=1)
        return x