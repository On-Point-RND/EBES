from functools import partial

import torch
import torch.nn.functional as F


def decay_cell(
    dt,  # (bs, 1)
    cell,  # (bs, hidden_size)
    cell_bar,  # (bs, hidden_size)
    decay_rate,  # (bs, hidden_size)
):
    return cell_bar + (cell - cell_bar) * torch.exp(-decay_rate * dt)


example_inputs = (
    torch.zeros(1, 1),
    torch.zeros(1),
    torch.zeros(1, 1),
    torch.zeros(1, 1),
    torch.zeros(1, 1),
    torch.zeros(1, 1),
    torch.zeros(7, 2),
    torch.zeros(7),
)


@partial(torch.jit.trace, example_inputs=example_inputs)
def ctlstm_cell(
    x,  # (bs, input_size)
    dt,  # (bs,)
    prev_out,  # (bs, hidden_size)
    prev_cell,  # (bs, hidden_size)
    prev_cell_bar,  # (bs, hidden_size)
    prev_decay_rate,  # (bs, hidden_size)
    weight,  # (7 * hidden_size, input_size + hidden_size)
    bias,  # (7 * hidden_size)
):
    # ll stands for left limit
    ll_cell = decay_cell(dt[:, None], prev_cell, prev_cell_bar, prev_decay_rate)
    ll_hidden = prev_out * torch.tanh(ll_cell)

    input_ = torch.cat((x, ll_hidden), dim=-1)
    output = F.linear(input_, weight, bias)
    (
        gate_input,
        gate_forget,
        gate_output,
        gate_pre_c,  # z
        gate_input_bar,
        gate_forget_bar,
        gate_decay,  # delta
    ) = output.chunk(7, dim=-1)
    gate_input = torch.sigmoid(gate_input)
    gate_forget = torch.sigmoid(gate_forget)
    gate_output = torch.sigmoid(gate_output)
    gate_pre_c = torch.tanh(gate_pre_c)
    gate_input_bar = torch.sigmoid(gate_input_bar)
    gate_forget_bar = torch.sigmoid(gate_forget_bar)
    decay_rate = F.softplus(gate_decay)

    cell = gate_forget * ll_cell + gate_input * gate_pre_c
    cell_bar = gate_forget_bar * prev_cell_bar + gate_input_bar * gate_pre_c
    return gate_output, cell, cell_bar, decay_rate
