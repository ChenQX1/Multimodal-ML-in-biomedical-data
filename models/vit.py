import torch
import torch.nn as nn

from .layers.vit.linear_projection import LinearProjection
from .layers.vit.vit_encoder import ViTEncoder


class MyViT(nn.Module):
    def __init__(self, input_shape, n_patches, hidden_d, h_heads, out_d, n_layers, k: int = 4, drop_p: float = 0.) -> None:
        super(MyViT, self).__init__()

        self.lp = LinearProjection(input_shape, n_patches, hidden_d)
        self.encoders = nn.ModuleList()
        for _ in range(n_layers):
            self.encoders.append(
                ViTEncoder(n_tokens=n_patches**2 + 1,
                           hidden_d=hidden_d, h_heads=h_heads, k=k, drop_p=drop_p)
            )
        self.clf = nn.Linear(hidden_d, out_d)

        self.model_args = {
            'input_shape': input_shape,
            'n_patches': n_patches,
            'hidden_d': hidden_d,
            'h_heads': h_heads,
            'out_d': out_d,
            'n_layers': n_layers
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.lp(x)
        for i, encoder in enumerate(self.encoders):
            tokens = encoder(tokens)
        logits = self.clf(tokens[:, 0])

        return logits

    def args_dict(self):

        return self.model_args
