"""Preprocessing model."""

from collections.abc import Mapping, Sequence
from typing import Literal

import torch
from torch import nn

from .basemodel import BaseModel
from ..types import Seq, Batch
from copy import deepcopy


class SeqBatchNorm(nn.Module):
    def __init__(self, num_count: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_count)

    def forward(self, x, lengths):  # (len, bs, num), (bs,)
        # (len, bs)
        len_mask = torch.arange(x.shape[0], device=x.device)[:, None] < lengths

        bx = x[len_mask]
        is_training = self.bn.training
        if is_training:
            self.bn(bx)  # update BN stats using only valid features (drop padding)

        self.bn.eval()
        # but compute actual BN on all values to assure that padding features stay equal
        # to the last valid
        res = self.bn(x.reshape(-1, x.shape[-1]))
        self.bn.train(is_training)  # return the state
        return res.reshape(x.shape)


class Batch2Seq(BaseModel):
    def __init__(
        self,
        cat_cardinalities: Mapping[str, int],
        num_count: int | None = None,
        num_features: Sequence[str] | None = None,
        cat_emb_dim: int | Mapping[str, int] | None = None,
        num_emb_dim: int | None = None,
        time_process: Literal["cat", "diff", "none"] = "none",
        num_norm: bool = False,
    ):
        super().__init__()
        cat_cardinalities = cat_cardinalities if cat_cardinalities is not None else {}

        if num_count is None:
            if num_features is not None:
                num_count = len(num_features)
            else:
                num_count = 0

        if time_process != "none":
            assert time_process in [
                "diff",
                "cat",
            ], "time_process may only be cat|diff|none"
            num_count += 1
        self._out_dim = 0

        self._cat_embs = nn.ModuleDict()
        cat_dims = []
        for name, card in cat_cardinalities.items():
            if cat_emb_dim is None:
                dim = int(min(600, round(1.6 * card**0.56)))
            elif isinstance(cat_emb_dim, int):
                dim = cat_emb_dim
            else:
                dim = cat_emb_dim[name]

            self._out_dim += dim
            cat_dims.append(dim)
            self._cat_embs[name] = nn.Embedding(card, dim)

        if num_emb_dim is None:
            if not cat_dims:
                raise ValueError(
                    "Auto dim choice for num embeddings does not work with no cat "
                    "features"
                )
            num_emb_dim = int(sum(cat_dims) / len(cat_dims))

        if num_count:
            self.batch_norm = SeqBatchNorm(num_count) if num_norm else None
            self._num_emb = nn.Conv1d(
                in_channels=num_count,
                out_channels=num_emb_dim * num_count,
                kernel_size=1,
                groups=num_count,
            )
        self._out_dim += num_emb_dim * num_count

    @property
    def output_dim(self):
        return self._out_dim

    def forward(self, batch: Batch) -> Seq:  # of shape (len, batch_size, )
        batch = deepcopy(batch)

        if not isinstance(batch.time, torch.Tensor):
            raise ValueError(
                "`time` field in batch must be a Tensor. "
                "Consider proper time preprocessing"
            )

        embs = []
        masks = []
        if batch.cat_features_names:
            for i, cf in enumerate(batch.cat_features_names):
                embs.append(self._cat_embs[cf](batch[cf]))
                if batch.cat_mask is not None:
                    mask = batch.cat_mask[:, :, i].unsqueeze(2)
                    mask = torch.repeat_interleave(
                        mask, self._cat_embs[cf].embedding_dim, 2
                    )
                    masks.append(mask)

        if batch.num_features is not None:
            assert self._num_emb
            x = batch.num_features
            if self.batch_norm:
                x = self.batch_norm(x, batch.lengths)
            x = x.permute(1, 2, 0)  # batch, features, len
            x = self._num_emb(x)
            embs.append(x.permute(2, 0, 1))
            if batch.num_mask is not None:
                masks.append(
                    torch.repeat_interleave(
                        batch.num_mask,
                        self._num_emb.out_channels // self._num_emb.in_channels,
                        dim=2,
                    )
                )

        tokens = torch.cat(embs, dim=2)
        masks = torch.cat(masks, dim=2) if len(masks) > 0 else None
        return Seq(tokens=tokens, lengths=batch.lengths, time=batch.time, masks=masks)
