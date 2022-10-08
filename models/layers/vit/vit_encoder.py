import torch
import torch.nn as nn

from .attention import MSA
from .attention import MHA


class ViTEncoder(nn.Module):
    def __init__(self, n_tokens, hidden_d, h_heads, k: int = 4, drop_p: float = 0.) -> None:
        super(ViTEncoder, self).__init__()

        self.n_tokens = n_tokens
        self.hidden_d = hidden_d

        self.ln1 = nn.LayerNorm((self.n_tokens, self.hidden_d))
        # self.msa = MSA(self.hidden_d, h_heads)
        self.msa = MHA(self.hidden_d, h_heads, drop_p)

        self.ln2 = nn.LayerNorm((self.n_tokens, self.hidden_d))
        self.enc_mlp = nn.Sequential(
            nn.Linear(self.hidden_d, self.hidden_d * k),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(hidden_d * k, hidden_d)
        )
    
    def forward(self, x: torch.Tensor):
        out = x + self.msa(self.ln1(x))   # residual connection
        out = out + self.enc_mlp(self.ln2(out))

        return out
