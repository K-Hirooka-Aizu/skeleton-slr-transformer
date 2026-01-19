import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,input_dim:int,head_dim:int,n_heads:int,bias=False):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.bias = bias

        self.linear_to_qkv = nn.Linear(input_dim,3*n_heads*head_dim,bias=self.bias)
        self.linear_w = nn.Linear(n_heads*head_dim,input_dim,bias=self.bias)

    def forward(self,x:torch.Tensor, mask=None) -> torch.Tensor:
        """
            inputs:
                x : (batch, * , seq_len, features)
            returns:
                z : the shape is same as input
                attn : self-attention weight (batch, * , seq_len, seq_len)
        """
        input_shape = x.size()
        
        qkv = self.linear_to_qkv(x)
        q,k,v = torch.chunk(qkv,3,dim=-1)
        
        q = q.view(input_shape[:-1]+(self.n_heads,self.head_dim)).transpose(-2,-3)
        k = k.view(input_shape[:-1]+(self.n_heads,self.head_dim)).transpose(-2,-3)
        v = v.view(input_shape[:-1]+(self.n_heads,self.head_dim)).transpose(-2,-3)

        attn = q@k.transpose(-1,-2) * self.head_dim**(-0.5)
        attn = F.softmax(attn,dim=-1)
        
        z = attn @ v
        z = self.linear_w(z.transpose(-2,-3).contiguous().view(input_shape[:-1]+(-1,)))
        
        return z, attn

class RelativePositionalEncodeMultiHeadSelfAttention(nn.Module):
    def __init__(self,input_dim:int,head_dim:int,n_heads:int,seq_len:int,bias=False):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.bias=bias

        relative_position_bias_table = nn.Parameter(torch.FloatTensor(2*seq_len-1,head_dim))
        self.register_buffer("relative_position_bias_table",relative_position_bias_table)
        torch.nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.linear_to_qkv = nn.Linear(input_dim,3*n_heads*head_dim,bias=self.bias)
        self.linear_w = nn.Linear(n_heads*head_dim,input_dim,bias=self.bias)


    def compute_relative_positions(self, seq_len):
        # Compute a matrix with relative positions between each pair of tokens
        # This returns indices that are used to retrieve relative embeddings
        range_vec = torch.arange(seq_len)
        rel_pos_matrix = range_vec[:, None] - range_vec[None, :]
        rel_pos_matrix = rel_pos_matrix + seq_len - 1  # shift to get positive indices
        return rel_pos_matrix

    def forward(self,x:torch.Tensor, mask=None) -> torch.Tensor:
        """
            inputs:
                x : (batch, * , seq_len, features)
            returns:
                z : the shape is same as input
                attn : self-attention weight (batch, * , seq_len, seq_len)
        """
        input_shape = x.size()
        
        qkv = self.linear_to_qkv(x)
        q,k,v = torch.chunk(qkv,3,dim=-1)
        
        q = q.view(input_shape[:-1]+(self.n_heads,self.head_dim)).transpose(-2,-3)
        k = k.view(input_shape[:-1]+(self.n_heads,self.head_dim)).transpose(-2,-3)
        v = v.view(input_shape[:-1]+(self.n_heads,self.head_dim)).transpose(-2,-3)

        
        pos_bias = self.relative_position_bias_table[self.compute_relative_positions(input_shape[-2])] 
        rel_attn_scores = torch.einsum('...hld,lrd->...hlr', q, pos_bias)

        attn = q@k.transpose(-1,-2) * self.head_dim**(-0.5)
        attn = F.softmax(attn+rel_attn_scores,dim=-1)
        
        z = attn @ v
        z = self.linear_w(z.transpose(-2,-3).contiguous().view(input_shape[:-1]+(-1,)))
        
        return z, attn
