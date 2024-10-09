# pyright: reportAttributeAccessIssue=false

from collections.abc import Iterable
from typing import TypeVar

import torch
import torch.nn.functional as F
import torcheval.metrics


TNegMeanSquaredError = TypeVar("TNegMeanSquaredError")
TPrimeNetAccuracy = TypeVar("TPrimeNetAccuracy", bound="PrimeNetAccuracy")


class PrimeNetAccuracy(torcheval.metrics.Metric):
    def __init__(self, *_, **__):
        super().__init__()
        self._add_state("correct", 0)
        self._add_state("total", 0)

    @torch.inference_mode()
    def update(self, pred: torch.Tensor, _):  # pyright: ignore
        self.correct += pred.correct_num
        self.total += pred.total_num
        return self

    @torch.inference_mode()
    def merge_state(self, metrics: Iterable[TPrimeNetAccuracy]):  # pyright: ignore
        for metric in metrics:
            self.correct += metric.correct
            self.total += metric.total
        return self

    @torch.inference_mode()
    def compute(self):
        return torch.tensor([self.correct / self.total])


class MultiLabelMeanAUROC(torcheval.metrics.BinaryAUROC):
    @torch.inference_mode()
    def update(  # pyright: ignore
        self,
        inp: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
    ):
        probas = F.sigmoid(inp).T
        target = target.T
        return super().update(probas, target, weight)

    @torch.inference_mode()
    def compute(self):
        return super().compute().mean()


class NegRootMeanSquaredError(torcheval.metrics.MeanSquaredError):
    @torch.inference_mode()
    def update(self, pred, target):  # pyright: ignore
        if pred.device != target.device:
            target = target.to(pred.device)
        self = super().update(pred, target)
        return self

    @torch.inference_mode()
    def compute(self: TNegMeanSquaredError) -> torch.Tensor:  # pyright: ignore
        return -1 * torch.sqrt(super().compute())


class LoggingMetric(torcheval.metrics.Metric):
    def __init__(self, key, *_, **__):
        super().__init__()
        self.key = key
        self._add_state("sum", 0)
        self._add_state("total", 0)

    @torch.inference_mode()
    def update(self, pred: torch.Tensor, _):  # pyright: ignore
        self.sum += (
            pred[self.key].item()
            if isinstance(pred[self.key], torch.Tensor)
            else pred[self.key]
        )
        self.total += 1
        return self

    @torch.inference_mode()
    def merge_state(self, metrics):  # pyright: ignore
        for metric in metrics:
            self.sum += metric.sum
            self.total += metric.total
        return self

    @torch.inference_mode()
    def compute(self):
        return torch.tensor([self.sum / self.total])


class MLEM_total_mse_loss(LoggingMetric):  # noqa: N801
    def __init__(self, *_, **__):
        super().__init__("total_mse_loss")


class MLEM_total_CE_loss(LoggingMetric):  # noqa: N801
    def __init__(self, *_, **__):
        super().__init__("total_CE_loss")


class MLEM_sparcity_loss(LoggingMetric):  # noqa: N801
    def __init__(self, *_, **__):
        super().__init__("sparcity_loss")


class MLEM_reconstruction_loss(LoggingMetric):  # noqa: N801
    def __init__(self, *_, **__):
        super().__init__("reconstruction_loss")
