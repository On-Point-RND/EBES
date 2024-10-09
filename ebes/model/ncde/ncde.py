import torch
from torch import nn
import torchcde

from ...types import Seq
from ..basemodel import BaseModel


class _FinalTanh(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        hidden_hidden_channels: int,
        num_hidden_layers: int,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = nn.Linear(hidden_channels, hidden_hidden_channels)
        self.linears = nn.ModuleList(
            nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
            for _ in range(num_hidden_layers - 1)
        )
        self.linear_out = nn.Linear(
            hidden_hidden_channels, input_channels * hidden_channels
        )

    def forward(self, _, z):
        z = self.linear_in(z)
        z = z.relu()
        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        z = self.linear_out(z).view(
            *z.shape[:-1], self.hidden_channels, self.input_channels
        )
        z = z.tanh()
        return z


class NCDE(BaseModel):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        hidden_hidden_size: int,
        num_hidden_layers: int,
    ):
        super().__init__()
        self.func = _FinalTanh(
            input_channels=input_size,
            hidden_channels=hidden_size,
            hidden_hidden_channels=hidden_hidden_size,
            num_hidden_layers=num_hidden_layers,
        )
        self.initial = nn.Linear(input_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)

    def forward(self, seq: Seq) -> torch.Tensor:
        inp = seq.tokens.permute(1, 0, 2)  # (bs, len, features)
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(inp)
        x = torchcde.CubicSpline(coeffs)
        x0 = x.evaluate(x.interval[0])
        z0 = self.initial(x0)
        zt = torchcde.cdeint(x, self.func, z0, x.interval, adjoint=False)
        assert isinstance(zt, torch.Tensor)
        last = zt[..., -1, :]
        last = self.bn(last)
        return last
