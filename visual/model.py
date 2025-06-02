# model.py
import torch
import torch.nn as nn
from torchvision.models import resnet18

class EmotionModel(nn.Module):
    def __init__(self, au_dim=17, num_classes=7):
        super(EmotionModel, self).__init__()
        self.cnn = resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()
        self.feat_dim = 512
        self.au_dim = au_dim
        self.input_dim = self.feat_dim + self.au_dim

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_dim, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(self.input_dim, num_classes)

    def forward(self, faces, aus):
        B, T, C, H, W = faces.shape
        faces = faces.view(B*T, C, H, W)
        feats = self.cnn(faces)  # [B*T, 512]
        aus = aus.view(B*T, -1)  # [B*T, 17]
        x = torch.cat([feats, aus], dim=1)  # [B*T, 512+17]
        x = x.view(B, T, -1).permute(1, 0, 2)  # [T, B, D]
        out = self.transformer(x)  # [T, B, D]
        out = out[-1]  # 最后一帧 [B, D]
        logits = self.classifier(out)
        return logits