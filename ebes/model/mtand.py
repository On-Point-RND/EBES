# https://github.com/mims-harvard/Raindrop/blob/main/code/baselines/mTAND/models.py

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..types import Seq
from .basemodel import BaseModel


class MultiTimeAttention(nn.Module):

    def __init__(self, input_dim, nhidden=16, embed_time=16, num_heads=1):
        super().__init__()

        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.linears = nn.ModuleList(
            [
                nn.Linear(embed_time, embed_time),
                nn.Linear(embed_time, embed_time),
                nn.Linear(input_dim * num_heads, nhidden),
            ]
        )

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)
        p_attn = F.softmax(scores, dim=-2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn * value.unsqueeze(-3), -2), p_attn

    def forward(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        batch, _, dim = value.size()
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = (
            lin(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key))
        )
        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.h * dim)
        return self.linears[-1](x)


class MTAND(BaseModel):

    def __init__(
        self,
        input_dim,
        nhidden=16,
        embed_time=16,
        num_heads=1,
    ):
        super().__init__()

        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.dim = input_dim
        self.nhidden = nhidden
        self.att = MultiTimeAttention(2 * input_dim, nhidden, embed_time, num_heads)
        self.periodic = nn.Linear(1, embed_time - 1)
        self.linear = nn.Linear(1, 1)

    def learn_time_embedding(self, tt):
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    @property
    def output_dim(self):
        return self.nhidden

    def forward(self, seq: Seq) -> Seq:
        assert seq.masks is not None
        time_steps = seq.time.T.contiguous()
        mask = seq.masks.permute(1, 0, 2).contiguous()
        x = torch.cat((seq.tokens.permute(1, 0, 2).contiguous(), mask), -1)
        mask = torch.cat((mask, mask), 2)

        key = self.learn_time_embedding(time_steps)
        q = torch.linspace(0, 1, 128, device=x.device)
        query = self.learn_time_embedding(q.unsqueeze(0))

        out = self.att(query, key, x, mask)
        out = out.permute(1, 0, 2)

        time = torch.linspace(0, 1, 128, device=x.device)
        lengths = torch.full(
            (out.shape[1],), out.shape[0], device=x.device, dtype=torch.int64
        )
        return Seq(tokens=out, lengths=lengths, time=time)
