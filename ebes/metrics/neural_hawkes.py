# pyright: reportAttributeAccessIssue=false

from typing import TypeVar
from collections.abc import Iterable

import torcheval
import torcheval.metrics
import torch

from ..types import NHReturn


TNHEventTypeAccuracy = TypeVar("TNHEventTypeAccuracy", bound="NHEventTypeAccuracy")


class NHEventTypeAccuracy(torcheval.metrics.Metric):
    def __init__(self, *_, **__):
        super().__init__()
        self._add_state("correct", 0)
        self._add_state("total", 0)

    @torch.inference_mode()
    def update(self, pred: NHReturn, _):  # pyright: ignore
        i_len = torch.arange(pred.clus_labels.shape[0], device=pred.clus_labels.device)
        hits = (pred.clus_labels == pred.pred_labels) & (i_len[:, None] < pred.lengths)
        self.correct += hits.sum()
        self.total += pred.lengths.sum()
        return self

    @torch.inference_mode()
    def merge_state(self, metrics: Iterable[TNHEventTypeAccuracy]):  # pyright: ignore
        for metric in metrics:
            self.correct += metric.correct
            self.total += metric.total
        return self

    @torch.inference_mode()
    def compute(self):
        return torch.tensor([self.correct / self.total])


class NHEventLogIntensity(torcheval.metrics.Mean):
    @torch.inference_mode()
    def update(self, pred: NHReturn, _):  # type: ignore
        max_len = pred.pre_event_intensities_of_gt.shape[0]
        device = pred.pre_event_intensities_of_gt.device
        i_len = torch.arange(max_len, device=device)[:, None]

        log_int_of_gt = torch.log(pred.pre_event_intensities_of_gt)
        log_int_of_gt_valid = log_int_of_gt[i_len < pred.lengths]
        return super().update(log_int_of_gt_valid)


class NHNegNonEventIntensity(torcheval.metrics.Mean):
    @torch.inference_mode()
    def update(self, pred: NHReturn, _):  # type: ignore
        return super().update(-pred.non_event_intensity)


class NHLL(torcheval.metrics.Mean):
    @torch.inference_mode()
    def update(self, pred: NHReturn, _):  # type: ignore
        max_len = pred.pre_event_intensities_of_gt.shape[0]
        device = pred.pre_event_intensities_of_gt.device
        i_len = torch.arange(max_len, device=device)[:, None]

        log_int_of_gt = torch.log(pred.pre_event_intensities_of_gt)
        log_int_of_gt_valid = log_int_of_gt[i_len < pred.lengths]
        ll = log_int_of_gt_valid.mean() - pred.non_event_intensity.mean()
        return super().update(ll)
