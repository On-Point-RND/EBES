import torch
from torch import nn
import torch.nn.functional as F

from .utils import decay_cell
from .ctlstm import CTLSTMCellCarry, CTLSTMCell
from ...types import NHReturn, NHSeq
from ..basemodel import BaseModel


class NeuralHawkes(BaseModel):
    _NODES_PER_EVENT = 5

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        bias: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.ctlstm_cell = CTLSTMCell(input_size, hidden_size, bias)
        self.projection = nn.Linear(hidden_size, output_size, bias=bias)

    def forward(self, seq: NHSeq) -> NHReturn:
        input_ = seq.tokens
        time = seq.time
        lengths = seq.lengths
        gts = seq.clus_labels

        batch_size = len(seq)
        device = input_.device
        dtype = input_.dtype

        carry = CTLSTMCellCarry.zeros(
            batch_size, self.hidden_size, device=device, dtype=dtype
        )

        max_dt = torch.clamp(
            (time[-1] - time[0]) / (lengths * self._NODES_PER_EVENT),
            min=torch.finfo(time.dtype).eps,
        )  # (bs,)

        intensity_integral = torch.zeros(batch_size, device=device, dtype=dtype)
        pre_event_intensities_of_gt = torch.empty(
            time.shape, device=device, dtype=dtype
        )
        pred_labels = torch.zeros(time.shape, device=device)

        prev_t = time[0]  # (bs,)
        for i, (inp, t, gt) in enumerate(zip(input_, time, gts)):

            ##### calculate intensity integral #####
            n_nodes: int = (
                torch.ceil((t - prev_t) / max_dt).clamp_(1).max().long()
            )  # type: ignore
            actual_dt = (t - prev_t) / n_nodes

            int_eval_nodes = (
                (
                    torch.linspace(0, 1, n_nodes + 1, device=device, dtype=dtype)[
                        :-1, None
                    ]
                )
                * (t - prev_t)
                + prev_t
                + actual_dt / 2  # midpoint rule
            )  # (nodes, bs)

            prev_cell_decayed = decay_cell(
                (int_eval_nodes - prev_t)[..., None],  # now (nodes, bs, 1)
                carry.cell,  # (bs, hidden_size)
                carry.cell_bar,  # (bs, hidden_size)
                carry.decay_rate,  # (bs, hidden_size)
            )  # (nodes, bs, hidden_size)
            hidden_decayed = carry.output * torch.tanh(prev_cell_decayed)
            intensity = F.softplus(
                self.projection(hidden_decayed)
            )  # (nodes, bs, output_size)
            intensity_integral += intensity.sum((0, 2)) * actual_dt

            ##### pre-event intensities #####

            # left limit, i.e. decayed to current time
            ll_cell = decay_cell(
                (t - prev_t)[..., None],  # now (bs, 1)
                carry.cell,  # (bs, hidden_size)
                carry.cell_bar,  # (bs, hidden_size)
                carry.decay_rate,  # (bs, hidden_size)
            )
            ll_hidden = carry.output * torch.tanh(ll_cell)
            # (bs, output_size)
            pre_event_int = F.softplus(self.projection(ll_hidden))
            pred_labels[i] = pre_event_int.argmax(1)
            i_batch = torch.arange(batch_size, device=device)
            pre_event_intensities_of_gt[i] = pre_event_int[i_batch, gt]

            ##### CTLSTM step #####
            carry = self.ctlstm_cell(inp, t - prev_t, carry)
            prev_t = t

        ret = NHReturn(
            pre_event_intensities_of_gt=pre_event_intensities_of_gt,
            non_event_intensity=intensity_integral,
            lengths=lengths,
            clustering_loss=seq.clustering_loss,
            clus_labels=seq.clus_labels,
            pred_labels=pred_labels,
        )
        return ret
