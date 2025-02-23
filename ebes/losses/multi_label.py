import logging
import torch
from torch import nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MultiLabelBinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probas = F.sigmoid(logits)
        if torch.isnan(probas).any():
            logger.warning("nn.BCELoss had RuntimeError, return nan")
            return torch.tensor(float("nan"))
        return self.bce(probas, target.float())
