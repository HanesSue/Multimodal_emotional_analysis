# -*- coding : utf-8 -*-
from transformers import WavLMProcessor
import torchaudio

class AudioProcessor:
    def __init__(self, model_name="microsoft/wavlm-base-plus"):
        self.processor = WavLMProcessor.from_pretrained(model_name)

    def process_audio(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        inputs = self.processor(
            waveform.squeeze(0).numpy(), 
            sampling_rate=sample_rate, 
            return_tensors="pt"
        )
        return inputs