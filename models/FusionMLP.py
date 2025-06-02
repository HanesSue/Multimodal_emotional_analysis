import torch
import torch.nn as nn
from config import (
    VISUAL_ENCODER_PATH,
    AUDIO_ENCODER_PATH,
    TEXT_ENCODER_PATH,
    VISUAL_FEATURE_DIM,
    AUDIO_FEATURE_DIM,
    TEXT_FEATURE_DIM,
    MLP_HIDDEN_DIM,
    FINETUNE_ENCODERS,
    DEVICE,
)
class FusionMLP(nn.Module):
    """
    将三个编码器各自输出的特征拼接在一起，再经过一层 MLP，输出最终的情感分值。
    """
    def __init__(self):
        super(FusionMLP, self).__init__()
        total_dim = VISUAL_FEATURE_DIM + AUDIO_FEATURE_DIM + TEXT_FEATURE_DIM
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(MLP_HIDDEN_DIM, 1)   # 最终输出一个标量情感分值
        )

    def forward(self, v_feat, a_feat, t_feat):
        # v_feat: (batch_size, VISUAL_FEATURE_DIM)
        # a_feat: (batch_size, AUDIO_FEATURE_DIM)
        # t_feat: (batch_size, TEXT_FEATURE_DIM)
        fusion = torch.cat([v_feat, a_feat, t_feat], dim=1)
        out = self.mlp(fusion)  # (batch_size, 1)
        return out.squeeze(1)   # 返回 (batch_size,)