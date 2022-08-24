import torch
import torch.nn as nn

from models.layers.tabnet.attentive_transformer import AttentiveTransformer
from models.layers.tabnet.ghost_bn import GBN


class GLU(nn.Module):
    def __init__(self, in_feats, out_feats, fc=None, vbs=128) -> None:
        super().__init__()

        if fc is not None:
            self.fc = fc
        else:
            self.fc = nn.Linear(in_feats, out_feats * 2)
        self.bn = GBN(out_feats * 2, vbs=vbs)
        self.out_feats = out_feats

    def forward(self, x):
        x = self.bn(self.fc(x))
        outputs = x[:, : self.out_feats] * torch.sigmoid(x[:, self.out_feats:])

        return outputs


class FeatureTransformer(nn.Module):
    def __init__(self, in_feats, out_feats, shared: nn.ModuleList, n_ind, vbs=128) -> None:
        super().__init__()

        first = True
        self.shared: nn.ModuleList = nn.ModuleList()
        if len(shared) != 0:
            self.shared.append(
                GLU(in_feats, out_feats, shared[0], vbs)
            )
            first = False
            for fc in shared[1:]:
                self.shared.append(
                    GLU(out_feats, out_feats, fc, vbs)
                )

        self.independent: nn.ModuleList = nn.ModuleList()
        if first:
            self.independent.append(
                GLU(in_feats, out_feats, vbs=vbs)
            )

        for _ in range(first, n_ind):
            self.independent.append(
                GLU(out_feats, out_feats, vbs=vbs)
            )
        self.scale = nn.parameter.Parameter(torch.sqrt(torch.tensor([0.5])), requires_grad=False)

    def forward(self, x):
        if len(self.shared) != 0:
            x = self.shared[0](x)
            for glu in self.shared[1:]:
                x = torch.add(x, glu(x))
                x = x * self.scale

        for glu in self.independent:
            x = torch.add(x, glu(x))
            x = x * self.scale

        return x


class Decision(nn.Module):
    def __init__(self, in_feats, n_d, n_a, shared, n_ind, relax, vbs=128) -> None:
        super().__init__()
        self.ft = FeatureTransformer(
            in_feats, n_d + n_a, shared, n_ind, vbs)
        self.at = AttentiveTransformer(n_a, in_feats, relax, vbs)

    def forward(self, x, a, prior):
        mask, prior = self.at(a, prior)
        x = self.ft(x * mask)
        sparse_loss = ((-1) * mask * torch.log(mask + 1e-10)).mean()

        return x, prior, sparse_loss

