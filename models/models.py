# -*- coding: utf-8 -*-
from torch import nn


class AudioExtractor:
    pass


class TextExtractor:
    pass


class VideoExtractor:
    pass


class MultiModalRegressor(nn.Module):
    def __call__(self, *args, **kwds):
        super().__call__(*args, **kwds)
        self.audio_extractor = AudioExtractor()
        self.text_extractor = TextExtractor()
        self.video_extractor = VideoExtractor()

    def forward(self, *args, **kwds):
        pass
