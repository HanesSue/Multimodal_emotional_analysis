import os
import torch
import torch.nn as nn
import numpy as np
from transformers import (
    HubertModel,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
from utils.Processor import MakeProcessor


class HuBertSentimentRegressor(nn.Module):
    def __init__(
        self,
        pretrained_model="",
    ):
        super().__init__()
        self.backbone = HubertModel.from_pretrained(pretrained_model)
        self.backbone.feature_extractor._freeze_parameters()  # 冻结特征提取器
        self.pooling = nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        hidden_size = self.backbone.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.1),
            nn.Linear(128, 1),  # 输出一个连续值
            nn.Tanh(),
        )

    def forward(self, input_values, attention_mask=None):
        outputs = self.backbone(
            input_values=input_values, attention_mask=attention_mask
        )  # 输入音频特征
        hidden_states = outputs.last_hidden_state  # (batch, time, hidden)
        x = self.pooling(hidden_states.transpose(1, 2))  # (batch, hidden, 1)
        # x = hidden_states.mean(dim=1)  # (batch, hidden)
        regression_output = self.regressor(x.squeeze(-1))  # (batch, 1)
        return regression_output.squeeze(-1)  # (batch,)


class AudioExtractor:
    def __init__(self, pretrained_model=None, details=True, **kwargs):
        self.details = details
        self.DEVICE = kwargs.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.EPOCHS = kwargs.get("epochs", 10)
        self.LR = kwargs.get("lr", 1e-4)
        self.MODEL_NAME = kwargs.get("model_name", "facebook/hubert-base-ls960")
        self.SAVE_PATH = kwargs.get(
            "save_path", "./checkpoints/WavLM_Emotion_Regression.pth"
        )
        os.makedirs(os.path.dirname(self.SAVE_PATH), exist_ok=True)

        if details:
            print(f"Device: {self.DEVICE}, Epochs: {self.EPOCHS}, LR: {self.LR}")
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

        self.model = HuBertSentimentRegressor(pretrained_model=self.MODEL_NAME).to(
            self.DEVICE
        )
        if pretrained_model is not None:
            self.model.load_state_dict(
                torch.load(pretrained_model, map_location=self.DEVICE)
            )
            if self.details:
                print(f"Loaded pretrained model from {pretrained_model}")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.LR, weight_decay=1e-4
        )
        self.criterion = nn.MSELoss()

    def discretize(self, value):
        values = np.array(self.DISCRETE_VALUES)
        return values[np.argmin(np.abs(values - value))]

    def predict(self, audio):
        self.model.eval()
        processor = MakeProcessor(mode="audio", model_name=self.MODEL_NAME)
        input_values = processor.processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_values.to(self.DEVICE)
        with torch.no_grad():
            output = self.model(input_values.to(self.DEVICE))
        return self.discretize(output.item())

    def train(self, train_loader, val_loader):
        total_steps = len(train_loader) * self.EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )
        self.model.train()

        for epoch in range(self.EPOCHS):
            epoch_loss = 0.0
            progress_bar = tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{self.EPOCHS}", leave=False
            )
            for batch in progress_bar:
                inputs, labels = batch
                inputs = {k: v.to(self.DEVICE) for k, v in inputs.items()}
                labels = labels.to(self.DEVICE).float()

                self.optimizer.zero_grad()

                outputs = self.model(**inputs)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})

            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{self.EPOCHS} - Avg Loss: {avg_loss:.4f}")
            val_loss = self.evaluate(val_loader)
            torch.save(self.model.state_dict(), self.SAVE_PATH)

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        progress_bar = tqdm(val_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for batch in progress_bar:
                inputs, labels = batch
                inputs = {k: v.to(self.DEVICE) for k, v in inputs.items()}
                labels = labels.to(self.DEVICE).float()

                outputs = self.model(**inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                progress_bar.set_postfix({"val_loss": loss.item()})

        avg_loss = total_loss / len(val_loader)
        print(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss
