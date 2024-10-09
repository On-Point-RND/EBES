from dataclasses import dataclass, replace
import math

import torch
import torch.nn as nn

from .utils import ctlstm_cell
from ..seq2seq import BaseSeq2Seq
from ...types import Seq


@dataclass
class CTLSTMCellCarry:
    output: torch.Tensor
    cell: torch.Tensor
    cell_bar: torch.Tensor
    decay_rate: torch.Tensor

    @classmethod
    def zeros(cls, batch_size: int, hidden_size: int, device=None, dtype=None):
        return cls(
            torch.zeros(batch_size, hidden_size, device=device, dtype=dtype),
            torch.zeros(batch_size, hidden_size, device=device, dtype=dtype),
            torch.zeros(batch_size, hidden_size, device=device, dtype=dtype),
            torch.zeros(batch_size, hidden_size, device=device, dtype=dtype),
        )


class CTLSTMCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight = nn.Parameter(
            torch.empty((hidden_size * 7, input_size + hidden_size))  # out, in
        )
        nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.empty(7 * hidden_size))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    def forward(
        self,
        input_: torch.Tensor,  # (bs, input_size)
        dt: torch.Tensor,  # (bs,)
        carry: CTLSTMCellCarry,
    ):
        carry_tuple = ctlstm_cell(
            input_,
            dt,
            carry.output,
            carry.cell,
            carry.cell_bar,
            carry.decay_rate,
            self.weight,
            self.bias,
        )
        carry = CTLSTMCellCarry(*carry_tuple)
        return carry


class CTLSTM(BaseSeq2Seq):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.ctlstm_cell = CTLSTMCell(input_size, hidden_size, bias)

    @property
    def output_dim(self):
        return self.hidden_size

    def forward(self, seq: Seq) -> Seq:
        dts = torch.diff(seq.time, dim=0, prepend=seq.time[0][None])
        carry = CTLSTMCellCarry.zeros(
            len(seq), self.hidden_size, device=seq.tokens.device, dtype=seq.tokens.dtype
        )

        hiddens = []
        for inp, dt in zip(seq.tokens, dts):
            carry = self.ctlstm_cell(inp, dt, carry)
            hidden = carry.output * torch.tanh(carry.cell)
            hiddens.append(hidden)

        return replace(seq, tokens=torch.stack(hiddens))
