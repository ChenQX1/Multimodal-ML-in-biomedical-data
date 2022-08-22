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


class PEElasticNet(PENetClassifier):
    def __init__(self, model_depth=50, cardinality=32, num_channels=3, num_classes=600, init_method=None, **kwargs):
        super(PEElasticNet, self).__init__(model_depth, cardinality,
                                           num_channels, num_classes, init_method, **kwargs)
        self.joint_pool = nn.AdaptiveAvgPool3d(1)

    def forward_feature(self, x):
        # Expand input (allows pre-training on RGB videos, fine-tuning on Hounsfield Units)
        if x.size(1) < self.num_channels:
            x = x.expand(-1, self.num_channels // x.size(1), -1, -1, -1)
        x = self.in_conv(x)
        x = self.max_pool(x)

        # Encoders
        for encoder in self.encoders:
            x = encoder(x)

        x = self.joint_pool(x)
        x = x.view(x.size(0), -1)

        return x
