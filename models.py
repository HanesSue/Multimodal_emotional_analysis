# models.py

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

class PretrainedEncoderWrapper(nn.Module):
    """
    简单封装一个预训练编码器：
    - 假设预训练模型的 forward 方法输入相应特征，输出一个固定维度特征向量或情感分值
    - 这里我们用 load_state_dict 加载 .pt 权重
    - 在训练时，可以通过 FINETUNE_ENCODERS 决定是否要 freeze
    """
    def __init__(self, model_path: str):
        super(PretrainedEncoderWrapper, self).__init__()
        # 实例化一个空的模型结构，与预训练时保持一致
        # 这里只给出示例：假设三个模型的结构均为简单的 nn.Sequential
        # 真实情况下，请替换为你实际训练时的模型定义
        # -------------------------------------------------------------------
        # 以视觉编码器为例：
        # self.model = nn.Sequential(
        #    nn.Conv3d(...),  # 或者 3D-ResNet、I3D、X3D 等
        #    ...
        #    nn.Linear(512, VISUAL_FEATURE_DIM)
        # )
        # -------------------------------------------------------------------
        # 为了示例，这里我们用一个 placeholder：一个全连接网络
        self.model = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, VISUAL_FEATURE_DIM),
        )

        # 加载预训练权重
        state_dict = torch.load(model_path, map_location=DEVICE)
        self.model.load_state_dict(state_dict)

        if not FINETUNE_ENCODERS:
            # 如果不微调，冻结所有参数
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        x: 输入特征张量，形状根据具体编码器而定
        返回: 特征向量 (batch_size, feature_dim)，或 (batch_size, 1) 的情感分
        """
        return self.model(x)


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
