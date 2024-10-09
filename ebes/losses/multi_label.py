import torch
from torch import nn
import torch.nn.functional as F


class MultiLabelBinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probas = F.sigmoid(logits)
        return self.bce(probas, target.float())
