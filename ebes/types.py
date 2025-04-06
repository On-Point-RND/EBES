from dataclasses import fields, dataclass
from typing import Any

import numpy as np
import torch


@dataclass(kw_only=True)
class Batch:
    lengths: torch.Tensor  # (batch,)
    time: np.ndarray | torch.Tensor  # (len, batch)
    index: torch.Tensor | np.ndarray | None = None  # (batch,)
    num_features: torch.Tensor | None = None  # (len, batch, features)
    cat_features: torch.Tensor | None = None  # (len, batch, features)
    target: torch.Tensor | None = None  # (batch,), (len, batch) or (batch, n_targets)
    cat_features_names: list[str] | None = None
    num_features_names: list[str] | None = None
    cat_mask: torch.Tensor | None = None  # (len, batch, features)
    num_mask: torch.Tensor | None = None  # (len, batch, features)

    def to(self, device: str):
        for field in fields(self):
            f = getattr(self, field.name)
            if isinstance(f, torch.Tensor):
                setattr(self, field.name, f.to(device))

        return self

    def _find_tensor_and_idx(self, feature_name):
        if self.cat_features is not None and self.cat_features_names is not None:
            try:
                cat_idx = self.cat_features_names.index(feature_name)
            except ValueError:
                pass
            else:
                return self.cat_features, cat_idx

        if self.num_features is not None and self.num_features_names is not None:
            try:
                num_idx = self.num_features_names.index(feature_name)
            except ValueError:
                pass
            else:
                return self.num_features, num_idx

        raise ValueError(
            f"Cannot access feature by name {feature_name}."
            f" Known cat names are: {self.cat_features_names}."
            f" Known num names are: {self.num_features_names}."
        )

    def __len__(self) -> int:
        """Batch size."""

        return len(self.lengths)

    def __setitem__(self, feature_name: str, value: Any):
        tensor, idx = self._find_tensor_and_idx(feature_name)
        tensor[:, :, idx] = value

    def __getitem__(
        self,
        feature_name: str,
    ) -> torch.Tensor:  # of shape (len, batch)
        tensor, idx = self._find_tensor_and_idx(feature_name)
        return tensor[:, :, idx]

    def __eq__(self, other):
        assert isinstance(other, self.__class__)
        equal = True
        for field in fields(self):
            my_field, other_field = getattr(self, field.name), getattr(
                other, field.name
            )
            if isinstance(my_field, torch.Tensor | np.ndarray):
                equal = equal and (my_field == other_field).all()
        return equal

    def pop_target(self) -> torch.Tensor | None:
        target = self.target
        self.target = None
        return target


@dataclass(kw_only=True)
class Seq:
    tokens: torch.Tensor  # of shape (len, batch, features)
    lengths: torch.Tensor  # of shape (batch,)
    time: torch.Tensor  # of shape (len, batch)
    masks: torch.Tensor | None = None  # of shape (len, batch, features, [2])

    def to(self, device):
        self.tokens = self.tokens.to(device)
        self.lengths = self.lengths.to(device)
        self.time = self.time.to(device)
        self.masks = self.masks.to(device) if self.masks else None
        return self

    def __len__(self):
        return len(self.lengths)


@dataclass(kw_only=True)
class NHSeq(Seq):
    clustering_loss: torch.Tensor  # of shape (len, batch)
    clus_labels: torch.Tensor  # of shape (len, batch)


@dataclass
class PrimeNetReturn:
    loss: torch.Tensor
    cl_loss: torch.Tensor
    mse_loss: torch.Tensor
    correct_num: int
    total_num: int

    def to(self, device: torch.device):
        self.loss = self.loss.to(device)
        self.cl_loss = self.cl_loss.to(device)
        self.mse_loss = self.mse_loss.to(device)
        return self

    def __getitem__(self, name):
        return self.__dict__[name]


@dataclass
class NHReturn:
    pre_event_intensities_of_gt: torch.Tensor  # (len, batch)
    non_event_intensity: torch.Tensor  # (batch,)
    clustering_loss: torch.Tensor  # of shape (len, batch)
    lengths: torch.Tensor  # (batch,)
    clus_labels: torch.Tensor  # of shape (len, batch)
    pred_labels: torch.Tensor  # of shape (len, batch)

    def to(self, device: torch.device):
        self.pre_event_intensities_of_gt = self.pre_event_intensities_of_gt.to(device)
        self.non_event_intensity = self.non_event_intensity.to(device)
        self.clustering_loss = self.clustering_loss.to(device)
        self.lengths = self.lengths.to(device)
        self.clus_labels = self.clus_labels.to(device)
        self.pred_labels = self.pred_labels.to(device)
        return self
