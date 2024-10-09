from torch.nn import CrossEntropyLoss, MSELoss
from .contrastive import ContrastiveLoss, InfoNCELoss
from .neural_hawkes import NHLoss
from .multi_label import MultiLabelBinaryCrossEntropyLoss
from .base import ModelLoss
