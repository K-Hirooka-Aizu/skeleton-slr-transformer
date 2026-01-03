import numpy as np
import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self,input_dim:int,input_len:int):
        super().__init__()

        PE = torch.empty((input_len,input_dim))
        
        for pos in range(input_len):
            for i in range(input_dim):
                if i%2==0:
                    PE[pos][i] = np.sin(pos/10000**(i/input_dim))
                else:
                    PE[pos][i] = np.cos(pos/10000**(i/input_dim))
        
        self.register_buffer("PE",PE)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return x + self.PE