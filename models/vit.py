import torch
import torch.nn as nn

from .layers.vit.linear_projection import LinearProjection
from .layers.vit.vit_encoder import ViTEncoder
from pytorch_pretrained_vit import ViT
from typing import Optional


class MyViT(nn.Module):
    def __init__(
            self, input_shape, n_patches, hidden_d, h_heads, out_d, n_layers,
            k: int = 4, drop_p: float = 0.) -> None:
        super(MyViT, self).__init__()

        self.lp = LinearProjection(input_shape, n_patches, hidden_d)
        self.encoders = nn.ModuleList()
        for _ in range(n_layers):
            self.encoders.append(
                ViTEncoder(
                    n_tokens=n_patches ** 2 + 1,
                    hidden_d=hidden_d, h_heads=h_heads, k=k,
                    drop_p=drop_p))
        self.clf = nn.Linear(hidden_d, out_d)

        self.model_args = {
            'input_shape': input_shape,
            'n_patches': n_patches,
            'hidden_d': hidden_d,
            'h_heads': h_heads,
            'out_d': out_d,
            'n_layers': n_layers,
            'k': k,
            'drop_p': drop_p
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.lp(x)
        for i, encoder in enumerate(self.encoders):
            tokens = encoder(tokens)
        logits = self.clf(tokens[:, 0])

        return logits

    def args_dict(self):

        return self.model_args


class ViTPytorch(ViT):
    def __init__(
            self, input_shape, n_patches, hidden_d, h_heads, out_d, n_layers,
            k: int = 4, drop_p: float = 0., transformer_name = 'B_32_imagenet1k') -> None:

        super().__init__(name=transformer_name, pretrained=True, patches=16, dim=hidden_d, ff_dim=3072, num_heads=h_heads,
                         num_layers=n_layers, attention_dropout_rate=0., dropout_rate=drop_p,
                         representation_size=None, load_repr_layer=False, classifier='token',
                         positional_embedding='1d', in_channels=3, image_size=384, num_classes=out_d)

        self.lp = LinearProjection(input_shape, n_patches, hidden_d)    # the output shape should be Bx145x768

        self.model_args = {
            'input_shape': input_shape,
            'n_patches': n_patches,
            'hidden_d': hidden_d,
            'h_heads': h_heads,
            'out_d': out_d,
            'n_layers': n_layers,
            'k': k,
            'drop_p': drop_p,
            'transformer_name': transformer_name
        }
    
    def forward(self, x):
        x = self.lp(x)

        x = self.transformer(x)  # b,gh*gw+1,d
        if hasattr(self, 'pre_logits'):
            x = self.pre_logits(x)
            x = torch.tanh(x)
        if hasattr(self, 'fc'):
            x = self.norm(x)[:, 0]  # b,d
            x = self.fc(x)  # b,num_classes
        
        return x
        
    def args_dict(self):
        return self.model_args