import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self,n_heads: int,d_embed :int,in_proj_bias=True,out_proj_bias=True):
        super().__init__()
        self.in_proj=nn.Linear(d_embed,3*d_embed,bias=in_proj_bias)
        self.out_proj=nn.Linear(d_embed,d_embed,bias=out_proj_bias)
        self.n_heads=n_heads
        self.d_head=d_embed//n_heads
    def forward(self,x:torch.Tensor,casual_mask=False)->torch.Tensor:
        # (batch_size,h, seq_len,dim/h)->
        input_shape=x.shape
        batch_size, sequence_length,d_embed=input_shape
        intermim_shape=(batch_size,sequence_length,self.n_heads,self.d_head)
        # (batch_size,seq_len,dim) -> (batch_size,seq_len,dim*3)
        q,k,v=self.in_proj(x).chunk(3,dim=-1)
        # x: (batch_size,seq_len,dim) -> (batch_size,seq_len,h,dim/h)->(batch_size,h,seq_len,dim/h)
        q=q.view(intermim_shape).transpose(1,2)
        k=k.view(intermim_shape).transpose(1,2)
        v=v.view(intermim_shape).transpose(1,2)
        weights=q@k.transpose(-1,-2)
        if casual_mask:
            # mask where is upper triangle (above the principal diagonal) is made up of 1
            mask=torch.ones_like(weights,dtype=torch.bool).triu(1)
            weights.masked_fill_(mask,-torch.inf)
        weights/=math.sqrt(self.d_head)
        weights=F.softmax(weights,dim=-1)
        # (batch_size,h,seq_len,seq_len)@(batch_size,h, seq_len,dim/h)->(batch_size,h,seq_len,dim/h)
        output=weights@v
        #(batch_size,h, seq_len,dim/h)->(batch_size,h, seq_len,dim/h)
        output=output.transpose(1,2)
        output=output.reshape(input_shape)
        output=self.out_proj(output)
        #(batch_size, seq_len,dim)
        return output


class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)  # Use d_cross here
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)  # Use d_cross here
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y) -> torch.Tensor:
        # x (latent): (batch_size, seq_len_q, d_embed)
        # y (context): (batch_size, seq_len_kv, d_cross)
        batch_size, seq_len_q, d_embed = x.shape
        seq_len_kv = y.shape[1]

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        # Separate heads for Q, K, and V
        q = q.view(batch_size, seq_len_q, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len_kv, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len_kv, self.n_heads, self.d_head).transpose(1, 2)

        # Attention calculation
        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)  # Softmax on the last dim

        output = weight @ v
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len_q, d_embed)

        return self.out_proj(output)