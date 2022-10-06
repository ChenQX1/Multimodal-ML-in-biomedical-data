import torch


def cyclical_encode(length: int, d: int) -> torch.Tensor:
    ans = torch.ones(length, d, dtype=torch.float)
    for i in range(length):
        for j in range(d):
            ans[i][j] = torch.sin(torch.tensor(i / (10000 ** (j / d))) ) if j % 2 == 0 else torch.cos(torch.tensor(i / (10000 ** ((j-1) / d))))
    
    return ans

