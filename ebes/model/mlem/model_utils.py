import torch.nn as nn
import torch
from ...types import Batch


class ReconPredictor(nn.Module):
    def __init__(
        self,
        dec_hidden_size,
        cat_cardinalities,
        num_features,
    ):
        super().__init__()

        self.cat_criterion = nn.CrossEntropyLoss(reduction="none")
        self.cat_cardinalities = cat_cardinalities or dict()
        self.cat_predictors = nn.ModuleDict()
        for name, vocab_size in self.cat_cardinalities.items():
            self.cat_predictors[name] = nn.Linear(dec_hidden_size, vocab_size)

        self.mse_fn = torch.nn.MSELoss(reduction="none")
        self.num_features = num_features or []
        self.num_predictors = nn.ModuleDict()
        for name in self.num_features:
            self.num_predictors[name] = nn.Linear(dec_hidden_size, 1)

    def forward(self, x_recon):
        predictions = {}
        for name in self.cat_cardinalities:
            predictions[name] = self.cat_predictors[name](x_recon)

        for i, name in enumerate(self.num_features):
            predictions[name] = self.num_predictors[name](x_recon)

        return predictions

    def loss(self, predictions: dict[str, torch.Tensor], batch: Batch):
        ce_loss = {}
        for name in self.cat_cardinalities:
            distribution = predictions[name]
            # We do this permutations for CrossEntropy
            labels = batch[name].long().permute(1, 0)
            assert batch.cat_mask is not None and batch.cat_features_names is not None
            mask = batch.cat_mask[:, :, batch.cat_features_names.index(name)].permute(
                1, 0
            )
            loss = self.cat_criterion(distribution.permute(1, 2, 0), labels) * mask
            ce_loss[name] = loss.sum(dim=1).mean()

        mse_loss = {}
        for name in self.num_features:
            pred = predictions[name].squeeze(-1)
            target = batch[name]
            assert batch.num_mask is not None and batch.num_features_names is not None
            mask = batch.num_mask[:, :, batch.num_features_names.index(name)]
            loss = self.mse_fn(pred, target) * mask
            mse_loss[name] = loss.sum(0).mean()

        return ce_loss, mse_loss
