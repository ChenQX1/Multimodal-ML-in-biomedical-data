import torch.nn as nn
from sparsemax import Sparsemax

from models.layers.tabnet.ghost_bn import GBN


class AttentiveTransformer(nn.Module):
    def __init__(self, n_a, in_feats, relax, vbs=128) -> None:
        super().__init__()
        self.fc = nn.Linear(n_a, in_feats)
        self.bn = GBN(in_feats, vbs=vbs)
        self.sparse_max = Sparsemax()
        self.r = relax

    def forward(self, a, prior):
        # a is the feature from the previous decision step
        a = self.bn(self.fc(a))
        mask = self.sparse_max(a * prior)
        prior = prior * (self.r - mask)

        return mask, prior
