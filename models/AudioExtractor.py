import torch
import torch.nn as nn
import numpy as np
from transformers import WavLMModel, WavLMConfig, get_linear_schedule_with_warmup

class AttentivePooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, x, mask=None):
        # x: (batch, time, dim)
        attn_weights = self.attention(x)  # (batch, time, 1)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        attn_weights = torch.softmax(attn_weights, dim=1)  # (batch, time, 1)
        weighted_sum = (x * attn_weights).sum(dim=1)  # (batch, dim)
        return weighted_sum


class WavLMSentimentRegressor(nn.Module):
    def __init__(
        self,
        pretrained_model="microsoft/wavlm-base-plus",
        freeze_feature_extractor=True,
    ):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained(pretrained_model)
        if freeze_feature_extractor:
            for param in self.wavlm.parameters():
                param.requires_grad = False

        hidden_size = self.wavlm.config.hidden_size
        self.pooling = AttentivePooling(hidden_size)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1),  # 输出一个连续值
        )

    def forward(self, input_values, attention_mask=None):
        outputs = self.wavlm(input_values=input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, time, hidden)
        pooled = self.pooling(hidden_states, mask=attention_mask)
        regression_output = self.regressor(pooled)
        return regression_output.squeeze(1)  # (batch,)


class AudioExtractor:
    def __init__(
        self, type="train", details=True, dataloader=None, **kwargs
    ):
        self.type = type
        self.details = details
        self.dataloader = dataloader
        self.DEVICE = kwargs.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.BATCH_SIZE = kwargs.get("batch_size", 32)
        self.EPOCHS = kwargs.get("epochs", 10)
        self.LR = kwargs.get("lr", 1e-4)
        self.MODEL_NAME = kwargs.get("model_name", "microsoft/wavlm-base-plus")
        self.SAVE_PATH = kwargs.get(
            "save_path", "./models/wavlm_sentiment_regressor.pth"
        )

        if details:
            print(
                f"AudioExtractor initialized with type: {self.type}, details: {self.details}"
            )
            print(
                f"Device: {self.DEVICE}, Batch Size: {self.BATCH_SIZE}, Epochs: {self.EPOCHS}, LR: {self.LR}"
            )
            print(f"Model Name: {self.MODEL_NAME}, Save Path: {self.SAVE_PATH}")

        self.DISCRETE_VALUES = [
            -1.0,
            -0.8,
            -0.6,
            -0.4,
            -0.2,
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
        ]

        self.model = WavLMSentimentRegressor(pretrained_model=self.MODEL_NAME).to(
            self.DEVICE
        )
        
        if self.type == "train":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.LR
            )
            self.criterion = nn.MSELoss()
            self.total_steps = len(self.dataloader) * self.EPOCHS
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=int(0.1 * self.total_steps),
                num_training_steps=self.total_steps,
            )
        
    def discretize(self, value):
        values = np.array(self.DISCRETE_VALUES)
        return values[np.argmin(np.abs(values - value))]
    
    def predict(self, audio):
        ...
        
    def train(self):
        ...