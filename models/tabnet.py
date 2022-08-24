import torch
import torch.nn as nn

from models.layers.tabnet.feature_transformer import Decision, FeatureTransformer


class TabNet(nn.Module):
    def __init__(
            self,
            in_feats,
            out_feats,
            n_d,
            n_a,
            n_shared,
            n_ind,
            n_steps,
            relax,
            vbs
            ) -> None:
        super().__init__()

        self.shared = nn.ModuleList()
        if n_shared > 0:
            self.shared.append(
                nn.Linear(in_feats, 2 * (n_d + n_a))
            )
            for _ in range(n_shared - 1):
                self.shared.append(
                    nn.Linear(n_d + n_a, 2 * (n_d + n_a))
                )

        self.first_step = FeatureTransformer(
            in_feats, out_feats=n_d + n_a, shared=self.shared, n_ind=n_ind, vbs=vbs)
        self.steps = nn.ModuleList()
        for _ in range(n_steps - 1):
            self.steps.append(
                Decision(in_feats, n_d, n_a, self.shared,
                         n_ind, relax, vbs)
            )
        self.fc = nn.Linear(n_d, out_feats)
        self.bn = nn.BatchNorm1d(in_feats, eps=0.1)
        self.n_d = n_d
        self.model_args = {
            'in_feats': in_feats,
            'out_feats': out_feats,
            'n_d': n_d,
            'n_a': n_a,
            'n_shared': n_shared,
            'n_ind': n_ind,
            'n_steps': n_steps,
            'relax': relax,
            'vbs': vbs,
        }

    def forward(self, x: torch.Tensor):
        device = next(self.parameters()).device
        x = self.bn(x)

        x_a = self.first_step(x)[:, self.n_d:]
        sparse_loss = torch.zeros(1, device=device)
        outputs = torch.zeros(x.size(0), self.n_d, device=device)
        prior = torch.ones(x.shape, device=device)
        for j, step in enumerate(self.steps):
            x_te, prior, l = step(x, x_a, prior)
            outputs += torch.relu(x_te[:, :self.n_d])
            x_a = x_te[:, self.n_d:]
            sparse_loss += l

        return self.fc(outputs), sparse_loss
    
    def args_dict(self):
        return self.model_args