import torch
from torch import nn

from ..types import NHReturn


class NHLoss(nn.Module):
    def forward(
        self,
        nh_return: NHReturn,
        _,
    ) -> torch.Tensor:
        max_len = nh_return.pre_event_intensities_of_gt.shape[0]
        device = nh_return.pre_event_intensities_of_gt.device
        i_len = torch.arange(max_len, device=device)[:, None]

        log_int_of_gt = torch.log(nh_return.pre_event_intensities_of_gt)
        log_int_of_gt_valid = log_int_of_gt[i_len < nh_return.lengths]
        nll = nh_return.non_event_intensity.mean() - log_int_of_gt_valid.mean()

        clus_loss = nh_return.clustering_loss[i_len < nh_return.lengths].mean()
        return nll + clus_loss
