from torch import nn


class ModelLoss(nn.Module):
    def forward(self, preds, _):
        return preds["loss"]
