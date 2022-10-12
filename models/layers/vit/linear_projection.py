import torch
import torch.nn as nn

from .attention import MSA
from .position_encoding import cyclical_encode


class LinearProjection(nn.Module):
    def __init__(self, input_shape, n_patches, hidden_d) -> None:
        super(LinearProjection, self).__init__()

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

        self.pos_encoding = nn.parameter.Parameter(cyclical_encode(self.n_patches ** 2 + 1, self.hidden_d))
    
    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        patches = x.reshape(b, self.n_patches ** 2, self.input_d)   # (b, # patches, patch dim)

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
        tokens += self.pos_encoding.repeat(b, 1, 1)

        return tokens
