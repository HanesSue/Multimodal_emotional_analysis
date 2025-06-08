import torch
import torch.nn as nn


class FusionMLP(nn.Module):
    def __init__(self):
        super(FusionMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        out = self.mlp(x)  # (batch_size, 1)
        return out.squeeze(1)  # 返回 (batch_size,)
