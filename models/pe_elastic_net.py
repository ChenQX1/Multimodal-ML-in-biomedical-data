import torch.nn as nn
from models.penet_classifier import PENetClassifier


class ElasticNet(nn.Module):
    def __init__(self, in_feats: int, out_feats: int = 1) -> None:
        super(ElasticNet, self).__init__()
        self.output_layer = nn.Sequential(
            nn.Linear(in_feats, in_feats * 2, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_feats * 2, out_feats, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.output_layer(x)

    def l1_reg(self):
        return self.output_layer[0].weight.abs().sum() + self.output_layer[2].weight.abs().sum()

    def l2_reg(self):
        return self.output_layer[0].weight.pow(2).sum() + self.output_layer[2].weight.pow(2).sum()


class PEElasticNet(PENetClassifier):
    def __init__(self, model_depth, cardinality=32, num_channels=3, num_classes=600, init_method=None, intermediate_n=12,**kwargs):
        super(PEElasticNet, self).__init__(model_depth, cardinality, num_channels, num_classes, init_method, **kwargs)
        self.output = nn.Linear(2048 * 2 * 6 * 6, intermediate_n)

    def forward(self, x):
        # Expand input (allows pre-training on RGB videos, fine-tuning on Hounsfield Units)
        if x.size(1) < self.num_channels:
            x = x.expand(-1, self.num_channels // x.size(1), -1, -1, -1)
        x = self.in_conv(x)
        x = self.max_pool(x)

        # Encoders
        for encoder in self.encoders:
            x = encoder(x)

        x = x.view(x.size(0), -1)
        x = self.output(x)

        return x
