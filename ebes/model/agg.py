"""Sequence to vector heads"""

from abc import ABC, abstractmethod

import torch

from .basemodel import BaseModel
from ..types import Seq


class BaseAgg(BaseModel, ABC):
    @abstractmethod
    def forward(self, seq: Seq) -> torch.Tensor: ...


class AllHiddenMean(BaseAgg):
    def forward(self, seq: Seq) -> torch.Tensor:
        return seq.tokens.mean(dim=0)


class TakeLastHidden(BaseAgg):
    def forward(self, seq: Seq) -> torch.Tensor:
        return seq.tokens[seq.lengths - 1, torch.arange(seq.tokens.shape[1])]


class ToTensor(BaseAgg):
    def forward(self, seq: Seq) -> torch.Tensor:
        return seq.tokens


class ValidHiddenMean(BaseAgg):
    def forward(self, seq: Seq) -> torch.Tensor:
        invalid = torch.zeros(
            seq.tokens.shape[0] + 1, *seq.tokens.shape[1:], device=seq.tokens.device
        )
        invalid[seq.lengths, torch.arange(invalid.shape[1])] = 1
        invalid = invalid.cumsum(dim=0).to(torch.bool)[:-1]
        return torch.where(invalid, torch.nan, seq.tokens).nanmean(dim=0)
