import torch
from torch import nn
import torch.nn.functional as F

from ...types import Seq, NHSeq
from ..basemodel import BaseModel


class NHClustering(BaseModel):
    def __init__(
        self,
        input_size: int,
        n_clusters: int,
        entropy_weight: float = 1.0,
        clustering_weight: float = 1.0,
        detach_input: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.n_clusters = n_clusters
        self.codebook = nn.Parameter(
            torch.empty(n_clusters, input_size), requires_grad=False
        )
        nn.init.normal_(self.codebook)
        self.entropy_weight = entropy_weight
        self.clustering_weight = clustering_weight
        self.detach_input = detach_input

    def forward(self, seq: Seq) -> NHSeq:
        inp = seq.tokens.detach() if self.detach_input else seq.tokens
        inp = F.normalize(inp)  # (len, bs, input_size)
        cb = F.normalize(self.codebook)  # (n_clusters, input_size)
        cos_sim = torch.inner(inp, cb)  # (len, bs, n_clusters)
        clus_labels = torch.argmax(cos_sim, dim=2)

        max_len, bs = seq.time.shape
        device = seq.time.device
        i_len = torch.arange(max_len, device=device)[:, None]
        i_batch = torch.arange(bs, device=device)

        log_p = F.log_softmax(cos_sim, 2)
        p = torch.exp(log_p)
        entropy = -torch.einsum("ijk,ijk->ij", p, log_p)

        log_loss = -log_p[i_len, i_batch, clus_labels]
        clustering_loss = (
            self.clustering_weight * log_loss - self.entropy_weight * entropy
        )

        return NHSeq(
            tokens=seq.tokens,
            lengths=seq.lengths,
            time=seq.time,
            masks=seq.masks,
            clus_labels=clus_labels,
            clustering_loss=clustering_loss,
        )
