import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import torch.nn.functional as F


class MSA(nn.Module):
    def __init__(self, d: int, h: int) -> None:
        super(MSA, self).__init__()

        self.d = d
        self.h = h

        assert d % h == 0, f"Cannot devide dim {d} into {h} heads."

        d_head = int(d / h)

        self.Q_mapping = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.h)])
        self.K_mapping = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.h)])
        self.V_mapping = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.h)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, seqs: torch.Tensor):
        # seq: (N, seq_length, token_dim)
        result = []
        for seq in seqs:
            seq_result = []
            for head in range(self.h):
                Q = self.Q_mapping[head]
                K = self.K_mapping[head]
                V = self.V_mapping[head]

                s = seq[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = Q(s), K(s), V(s)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))

        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


class MHA(nn.Module):
    def __init__(self, d: int, h: int, drop_p: float = 0.) -> None:
        super(MHA, self).__init__()

        self.d = d
        self.h = h

        assert d % h == 0, f"Cannot devide dim {d} into {h} heads."

        self.qkv_proj = nn.Linear(self.d, 3 * self.d)
        self.o_proj = nn.Linear(self.d, self.d)
        self.drop_layer = nn.Dropout(drop_p)

        # self.attn_layer = nn.MultiheadAttention(self.d, self.h, dropout=drop_p, batch_first=True)

    def forward(self, x, mask=None):
        # qkv = rearrange(self.qkv_proj(x), "b n (d qkv) -> (qkv) b n d", d=self.d, qkv=3)
        # queries, keys, values = qkv[0], qkv[1], qkv[2]
        # out, _ = self.attn_layer(queries, keys, values, attn_mask=mask)

        qkv = rearrange(self.qkv_proj(x), "b n (h d qkv) -> (qkv) b h n d", h=self.h, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        
        scaling = self.d ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.drop_layer(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values) # sum over the third axis
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.o_proj(out)
        
        return out
