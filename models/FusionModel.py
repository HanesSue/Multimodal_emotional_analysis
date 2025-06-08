from .AudioExtractor import AudioExtractor
from .TextExtractor import TextExtractor
from .VideoExtractor import VideoExtractor
from utils.Processor import MakeProcessor

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


class FusionExtractor:
    def __init__(self, **kwargs):
        audio_weights = kwargs.get("audio_weights")
        text_weights = kwargs.get("text_weights")
        video_weights = kwargs.get("video_weights")
        mlp_weights = kwargs.get("mlp_weights")
        self.audio_extractor = AudioExtractor(pretrained_model=audio_weights, details=True)
        self.text_extractor = TextExtractor(pretrained_model=text_weights, details=True)
        self.video_extractor = VideoExtractor()
        self.fusion_mlp = FusionMLP()
        self.fusion_mlp.load_state_dict(torch.load(mlp_weights))

    def predict(self, data, **kwargs):
        audio_processor = MakeProcessor(mode="audio", model_name="facebook/hubert-base-ls960")
        text_processor = MakeProcessor(mode="text", model_name="google-bert/bert-base-chinese")
        audio_features = self.audio_extractor.predict(audio, **kwargs)
        text_features = self.text_extractor.predict(text, **kwargs)
        video_features = self.video_extractor.predict(video, **kwargs)
        features = torch.cat([audio_features, text_features, video_features], dim=1)
        return self.fusion_mlp(features)