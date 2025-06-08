# -*- coding : utf-8 -*-
from transformers import Wav2Vec2FeatureExtractor, BertTokenizerFast
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import subprocess
import os


class MakeProcessor:
    def __init__(self, mode="audio", model_name="facebook/hubert-base-ls960"):
        """
        预处理器初始化，为Transformers模型创建预处理器
        Args:
            mode (str): _description_. 根据 mode 的不同，选择不同的预处理器。
                        "audio" 使用 Wav2Vec2FeatureExtractor 处理音频数据，
                        "text" 使用 BertTokenizerFast 处理文本数据。
            model_name (str, optional): _description_. 选择与Transformers模型对应的预处理器。
                                    默认为 "facebook/hubert-base-ls960"（音频）或 "bert-base-chinese"（文本）。

        """
        self.mode = mode
        self.model_name = model_name
        if self.mode == "audio":
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
        elif self.mode == "text":
            self.processor = BertTokenizerFast.from_pretrained(self.model_name)
        else:
            raise ValueError("mode must be one of audio or text")

    def process(self, data, **kwargs):
        """
        特征提取

        Args:
            data (Any): _输入数据，音频或文本。_
            sampling_rate (int, optional): _音频采样率，默认为16000。_
            max_length (int, optional): _文本最大长度，默认为64。_

        Returns:
            features (dict): _处理后的特征以及注意力掩码。_
        """
        if self.mode == "audio":
            sampling_rate = kwargs.get("sampling_rate", 16000)
            return self.processor(
                data,
                sampling_rate=sampling_rate,
                return_tensors="pt",
                padding="max_length",
                return_attention_mask=True,
                do_normalize=True,
                max_length=96000,
                truncation=True,
            )
        elif self.mode == "text":
            max_length = kwargs.get("max_length", 64)
            return self.processor(
                data,
                return_tensors="pt",
                padding="max_length",
                max_length=max_length,
                truncation=True,
            )
        else:
            raise ValueError("mode must be one of audio or text")