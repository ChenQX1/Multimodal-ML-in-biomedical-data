import torch
import torch.nn as nn


class GBN(nn.Module):
    def __init__(self, in_feats, vbs=64, momentum=0.01) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(in_feats, momentum=momentum)
        self.vbs = vbs

    def forward(self, x):
        chunk_size = x.size(0) // self.vbs
        chunk = torch.chunk(x, chunk_size, 0)
        res = [self.bn(y) for y in chunk]

        return torch.concat(res, 0)
