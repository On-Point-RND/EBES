from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def outer_pairwise_distance(a, b=None):
    """
    Compute pairwise_distance of Tensors
        A (size(A) = n x d, where n - rows count, d - vector size) and
        B (size(A) = m x d, where m - rows count, d - vector size)
    return matrix C (size n x m), such as
        C_ij = distance(i-th row matrix A, j-th row matrix B)

    if only one Tensor was given, computer pairwise distance to itself (B = A)
    """

    if b is None:
        b = a

    max_size = 2**26
    n = a.size(0)
    m = b.size(0)
    d = a.size(1)

    if n * m * d <= max_size or m == 1:
        return torch.pairwise_distance(
            a[:, None].expand(n, m, d).reshape((-1, d)),
            b.expand(n, m, d).reshape((-1, d)),
        ).reshape((n, m))

    else:
        batch_size = max(1, max_size // (n * d))
        batch_results = []
        for i in range((m - 1) // batch_size + 1):
            id_left = i * batch_size
            id_rigth = min((i + 1) * batch_size, m)
            batch_results.append(outer_pairwise_distance(a, b[id_left:id_rigth]))

        return torch.cat(batch_results, dim=1)


##### PAIR SELECTORS #####


class PairSelector(ABC):
    """Strategy to sample positive and negative embedding pairs."""

    @abstractmethod
    def get_pairs(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class HardNegativePairSelector(PairSelector):
    """
    Generates all possible possitive pairs given labels and
         neg_count hardest negative example for each example
    """

    def __init__(self, neg_count=1):
        super().__init__()
        self.neg_count = neg_count

    def get_pairs(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # construct matrix x, such as x_ij == 0 <==> labels[i] == labels[j]
        n = len(labels)
        x = labels - labels[:, None]  # The same but works for numpy. I've tested!!!

        # positive pairs
        positive_pairs = torch.triu((x == 0).int(), diagonal=1).nonzero(as_tuple=False)

        # hard negative minning
        mat_distances = outer_pairwise_distance(
            embeddings.detach()
        )  # pairwise_distance

        upper_bound = int((2 * n) ** 0.5) + 1
        mat_distances = (upper_bound - mat_distances) * (x != 0).type(
            mat_distances.dtype
        )  # filter: get only negative pairs

        _, indices = mat_distances.topk(k=self.neg_count, dim=0, largest=True)
        negative_pairs = torch.stack(
            [
                torch.arange(0, n, dtype=indices.dtype, device=indices.device).repeat(
                    self.neg_count
                ),
                torch.cat(indices.unbind(dim=0)),
            ]
        ).t()

        return positive_pairs, negative_pairs


##### LOSSES #####


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss

    "Signature verification using a siamese time delay neural network", NIPS 1993
    https://papers.nips.cc/paper/769-signature-verification-using-a-siamese-time-delay-neural-network.pdf
    """

    def __init__(
        self,
        margin: float,
        pair_selector: PairSelector,
    ):
        super().__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(
            embeddings, target
        )
        positive_loss = F.pairwise_distance(
            embeddings[positive_pairs[:, 0]], embeddings[positive_pairs[:, 1]]
        ).pow(2)

        negative_loss = F.relu(
            self.margin
            - F.pairwise_distance(
                embeddings[negative_pairs[:, 0]],
                embeddings[negative_pairs[:, 1]],
            )
        ).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)

        return loss.sum()


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss https://arxiv.org/abs/1807.03748
    """

    def __init__(
        self,
        temperature: float,
        pair_selector: PairSelector,
        angular_margin: float = 0.0,  # = 0.5 ArcFace default
    ):
        super().__init__()

        self.temperature = temperature
        self.pair_selector = pair_selector
        self.angular_margin = angular_margin

    def forward(self, embeddings, target):
        embeddings = self.project(embeddings)

        positive_pairs, _ = self.pair_selector.get_pairs(embeddings, target)
        dev = positive_pairs.device
        all_idx = torch.arange(len(positive_pairs), dtype=torch.long, device=dev)
        mask_same = torch.zeros(len(positive_pairs), len(embeddings), device=dev)
        mask_same[all_idx, positive_pairs[:, 0]] -= torch.inf

        sim = (
            F.cosine_similarity(
                embeddings[positive_pairs[:, 0], None],
                embeddings[None],
                dim=-1,
            )
            + mask_same
        )
        if self.angular_margin > 0.0:
            with torch.no_grad():
                target_sim = sim[all_idx, positive_pairs[:, 1]].clamp(0, 1)
                target_sim.arccos_()
                target_sim += self.angular_margin
                target_sim.cos_()
                sim[all_idx, positive_pairs[:, 1]] = target_sim

        sim /= self.temperature
        lsm = -F.log_softmax(sim, dim=-1)
        loss = torch.take_along_dim(
            lsm,
            positive_pairs[:, [1]],
            dim=1,
        ).sum()

        return loss
