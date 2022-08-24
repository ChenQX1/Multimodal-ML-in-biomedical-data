import torch
import torch.nn as nn


class FusionModel(nn.Module):
    def __init__(
        self,
        subnet_ct: nn.Module,
        subnet_ehr: nn.Module,
        shim_ct: nn.Module = None,
        shim_ehr: nn.Module = None,
        classifier_head: nn.Module = None) -> None:
        super(FusionModel, self).__init__()

        self.subnet_ct = subnet_ct
        self.subnet_ehr = subnet_ehr
        self.shim_ct = shim_ct
        self.shim_ehr = shim_ehr
        self.classifier_head = classifier_head
        if self.classifier_head:
            print(" ====== Joint Fusion ======")
        else:
            print(" ====== Post Fusion =======")

    def forward(self, feat_ct: torch.Tensor, feat_ehr: torch.Tensor) -> torch.Tensor:
        if self.classifier_head:
            feat_ct = self.subnet_ct.forward_feature(feat_ct)
            feat_ehr = self.subnet_ehr.forward_feature(feat_ehr)
            if self.shim_ct:
                feat_ct = self.shim_ct(feat_ct)
                feat_ct = feat_ct.view(feat_ct.size(0), -1)
            if self.shim_ehr:
                feat_ehr = self.shim_ehr(feat_ehr)
            joint_input = torch.concat([feat_ct, feat_ehr], dim=1)
            outputs = self.classifier_head(joint_input)
        else:
            logits_ct = self.subnet_ct(feat_ct)
            logits_ehr = self.subnet_ehr(feat_ehr)
            outputs = (logits_ct + logits_ehr) * 0.5
        
        return outputs
    
    def args_dict(self):
        model_args = {
            'subnet_ct': self.subnet_ct,
            'subnet_ehr': self.subnet_ehr,
            'shim_ct': self.shim_ct,
            'shim_ehr': self.shim_ehr,
            'classifier_head': self.classifier_head
        }

        return model_args