import torch
import torch.nn as nn

from .attention import MSA
from .position_encoding import cyclical_encode


# one-layer ViT
class ViT(nn.Module):
    def __init__(self, input_shape, n_patches, hidden_d, h_heads, out_d) -> None:
        super(ViT, self).__init__()

        self.input_shape = input_shape  # (C, H, W)
        self.n_patches = n_patches  # n_patches x n_patches
        self.patch_size = (
            input_shape[1] / n_patches, input_shape[2] / n_patches)
        self.input_d = int(
            input_shape[0] * self.patch_size[0] * self.patch_size[1])
        self.hidden_d = hidden_d

        assert input_shape[1] % n_patches == 0, f"Input dim 2 not entirely divisible by number of patches"
        assert input_shape[2] % n_patches == 0, f"Input dim 3 not entirely divisible by number of patches"

        self.linear_map = nn.Linear(self.input_d, self.hidden_d)
        self.class_token = nn.parameter.Parameter(torch.rand(1, self.hidden_d))
        self.ln1 = nn.LayerNorm((self.n_patches ** 2 + 1, self.hidden_d))
        self.msa = MSA(self.hidden_d, h_heads)

        self.ln2 = nn.LayerNorm((self.n_patches ** 2 + 1, self.hidden_d))
        self.enc_mlp = nn.Sequential(
            nn.Linear(self.hidden_d, self.hidden_d),
            nn.ReLU()
        )

        self.clf = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )

        self.pos_encoding = nn.parameter.Parameter(cyclical_encode(self.n_patches ** 2 + 1, self.hidden_d))

    def forward(self, x: torch.Tensor, pos_encoding: torch.Tensor):
        n, c, h, w = x.shape
        patches = x.reshape(n, self.n_patches ** 2, self.input_d)   # (n, # patches, patch dim)

        tokens = self.linear_map(patches)
        # classification token
        tokens = torch.stack(
            [torch.vstack((self.class_token, tokens[i]))
            for i in range(len(tokens))]
        )
        # positional encoding
        # pos_encoding = torch.from_numpy(
        #     pos_encode(self.n_patches ** 2 + 1, self.hidden_d)
        # ).repeat(n, 1, 1)
        tokens += self.pos_encoding.repeat(n, 1, 1)

        out = tokens + self.msa(self.ln1(tokens))   # residual connection

        out = out + self.enc_mlp(self.ln2(out))

        out = self.clf(out[:, 0])

        return out
