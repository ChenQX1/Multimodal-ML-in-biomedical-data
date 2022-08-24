import torch.nn as nn
from .penet_classifier import PENetClassifier


class ElasticNet(nn.Module):
    def __init__(self, in_feats: int, out_feats: int = 1) -> None:
        super(ElasticNet, self).__init__()
        self.normalizer = nn.BatchNorm1d(in_feats)
        self.backbone = nn.Sequential(
            nn.Linear(in_feats, in_feats, bias=True),
            nn.LeakyReLU(),
        )
        self.classifier_head = nn.Linear(in_feats, out_feats)

        self.model_args = {
            'in_feats': in_feats,
            'out_feats': out_feats
        }

    def forward(self, x):
        x = self.normalizer(x)
        x = self.backbone(x)
        x = self.classifier_head(x)

        return x

    def forward_feature(self, x):
        x = self.normalizer(x)
        x = self.backbone(x)

        return x

    def l1_reg(self):
        return self.backbone[0].weight.abs().sum() + self.backbone[2].weight.abs().sum()

    def l2_reg(self):
        return self.backbone[0].weight.pow(2).sum() + self.backbone[2].weight.pow(2).sum()

    def args_dict(self):
        return self.model_args
