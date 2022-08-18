import torch
import torch.nn as nn

"""
How to use the module:
model = BaseJointModel()
dataloader = MultimodalLoder()
batch = (ct_volume, ehr_feat, target)

when training sub-models individually:
optimizer = optimizer_fn(model.penet.parameters())
preds = model.penet(batch[0])
loss = criterion(preds, batch[2])
optimizer.zero_grad()
loss.backward()
optimizer.step()
...

when training jointly:
optimizer = optimizer_fn(model.parameters())
preds = model(batch[0], batch[1])
loss = criterion(preds, batch[2])
optimizer.zero_grad()
loss.backward()
optimizer.step()

For compatibility with the original PENet project,
we save and load submodels parameters indivisually.
"""

class PENetElasticNet(nn.Module):
    def __init__(
        self,
        subnet_ct: nn.Module,
        subnet_ehr: nn.Module,
        shim_ct: nn.Module,
        shim_ehr: nn.Module,
        classifier_head: nn.Module) -> None:

        self.subnet_ct = subnet_ct
        self.subnet_ehr = subnet_ehr
        self.shim_ct = shim_ct
        self.shim_ehr = shim_ehr
        self.classifier_head = classifier_head

    def forward(self, feat_ct: torch.Tensor, feat_ehr: torch.Tensor) -> torch.Tensor:
        feat_ct = self.subnet_ct.forward_feature(feat_ct)
        feat_ehr = self.subnet_ehr.forward_feature(feat_ehr)
        if self.shim_ct:
            feat_ct = self.shim_ct(feat_ct)
            feat_ct = feat_ct.view(feat_ct.size(0), -1)
        if self.shim_ehr:
            feat_ehr = self.shim_ehr(feat_ehr)
        joint_input = torch.concat([feat_ct, feat_ehr], dim=1)
        outputs = self.classifier_head(joint_input)

        return outputs
    
