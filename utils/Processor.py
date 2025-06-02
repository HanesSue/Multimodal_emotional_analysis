# -*- coding : utf-8 -*-
from transformers import Wav2Vec2FeatureExtractor, BertTokenizerFast
import torchaudio

class MakeProcessor:
    def __init__(self, mode="audio", model_name="facebook/wav2vec2-base-960h"):
        self.mode = mode
        self.model_name = model_name
        if self.mode == "audio":
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
        elif self.mode == "text":
            self.processor = BertTokenizerFast.from_pretrained(self.model_name)
        else:
            raise ValueError("mode must be one of audio or text")
        
    def process(self, data, **kwargs):
        if self.mode == "audio":
            sampling_rate = kwargs.get("sampling_rate", 16000)
            return self.processor(data, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        elif self.mode == "text":
            return self.processor(data, return_tensors="pt")
        else:
            raise ValueError("mode must be one of audio or text")
    