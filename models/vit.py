import torch
import torch.nn as nn

from .layers.vit.linear_projection import LinearProjection
from .layers.vit.vit_encoder import ViTEncoder
from vit_pytorch import ViT


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


class ViTPyTorch(ViT):
    def __init__(self, input_shape, n_patches, hidden_d, h_heads, out_d, n_layers, k: int = 4, drop_p: float = 0.) -> None:
        c, h, w = input_shape
        assert h == w, 'H and W are not equal!'
        assert input_shape[1] % n_patches == 0, 'The shape of image cannot be divided by n_patches'
        patch_size = int(input_shape[1] / n_patches)

        super().__init__(image_size=h, patch_size=patch_size, num_classes=out_d, dim=hidden_d, depth=n_layers, heads=h_heads, mlp_dim=hidden_d*k, pool='cls', channels=c, dropout=drop_p)

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

    def args_dict(self):
        return self.model_args