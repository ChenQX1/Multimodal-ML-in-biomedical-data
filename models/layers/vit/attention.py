import torch
import torch.nn as nn


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
