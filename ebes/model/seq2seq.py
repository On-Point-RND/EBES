"""Collection of Seq-2-Seq models"""

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Literal

import math
import torch
from torch import nn

from .basemodel import BaseModel
from ..types import Seq


class BaseSeq2Seq(BaseModel, ABC):
    @abstractmethod
    def forward(self, seq: Seq) -> Seq: ...


class Projection(BaseSeq2Seq):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()
        self._out_dim = out_features
        self.linear = nn.Linear(in_features, out_features, bias)

    @property
    def output_dim(self):
        return self._out_dim

    def forward(self, seq: Seq) -> Seq:
        return replace(seq, tokens=self.linear(seq.tokens))


class GRU(BaseSeq2Seq):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
        initial_hidden: Literal["default", "static"] = "static",
    ):
        super().__init__()
        d = 2 if bidirectional else 1
        self._out_dim = hidden_size * d

        self.net = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        self.hx = None
        if initial_hidden == "static":
            self.hx = nn.Parameter(torch.randn(d * num_layers, 1, hidden_size))

    @property
    def output_dim(self):
        return self._out_dim

    def forward(self, seq: Seq) -> Seq:
        hx = self.hx
        if hx is not None:
            hx = hx.expand(-1, len(seq), -1).contiguous()

        x, _ = self.net(seq.tokens, hx)
        return Seq(tokens=x, lengths=seq.lengths, time=seq.time)


class PositionalEncoding(nn.Module):
    """
    https://github.com/pytorch/examples/blob/main/word_language_model/model.py

    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000, enc_type="base"):
        super().__init__()
        self.enc_type = enc_type
        self.dropout = nn.Dropout(p=dropout)
        if enc_type == "base":
            pe = self.get_pe(max_len, d_model)
            self.register_buffer("pe", pe)
        elif enc_type == "cat":
            pe = self.get_pe(max_len, 16)
            self.register_buffer("pe", pe)
        elif enc_type == "learned":
            self.pe = nn.Parameter(torch.randn(max_len, 1, d_model))
        elif enc_type == "none":
            self.pe = torch.zeros((max_len, 1, d_model))
        else:
            raise NotImplementedError(
                f"Do not support {enc_type} type! "
                f"Only support ['none', 'base', 'learned']"
            )

    def get_pe(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[-(d_model % 2) :])
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        self.pe = self.pe.to(x.device)
        if self.enc_type == "cat":
            x = torch.cat([x, self.pe[: x.size(0)].expand(-1, x.shape[1], -1)], dim=2)
        else:
            x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class Transformer(BaseSeq2Seq):
    def __init__(
        self,
        input_size: int,
        max_len: int,
        num_layers: int = 1,
        num_heads: int = 1,  # Dont change so we dont have to worry about input_size
        scale_hidden: int = 4,
        dropout: float = 0.0,
        pos_dropout: float = 0.0,
        pos_enc_type: str = "base",
    ):
        super().__init__()
        self.pos_encoder = PositionalEncoding(
            input_size, pos_dropout, max_len, pos_enc_type
        )
        if pos_enc_type == "cat":
            input_size += 16
        self._out_dim = input_size
        # Transformer encoder with multiple layers
        encoder_layer = nn.TransformerEncoderLayer(
            input_size,
            num_heads,
            dim_feedforward=input_size * scale_hidden,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers,
        )

    @property
    def output_dim(self):
        return self._out_dim

    def forward(self, seq: Seq) -> Seq:
        src = seq.tokens
        src = self.pos_encoder(src)
        padding_mask = (
            torch.arange(src.shape[0], device=src.device) >= seq.lengths[:, None]
        )
        encoded_src = self.transformer_encoder(src, src_key_padding_mask=padding_mask)
        return Seq(tokens=encoded_src, lengths=seq.lengths, time=seq.time)
