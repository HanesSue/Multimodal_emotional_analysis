# -*- coding : utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import os
from transformers import BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
from utils.Processor import MakeProcessor


# 定义BertTextCNN回归模型
class BertTextCNNRegressor(nn.Module):
    def __init__(
        self,
        pretrained_model_name,
        embed_dim=768,
        num_filters=128,
        filter_sizes=[3, 4, 5],
    ):
        super(BertTextCNNRegressor, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim, out_channels=num_filters, kernel_size=fs
                )
                for fs in filter_sizes
            ]
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(num_filters * len(filter_sizes), 1)

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        last_hidden_state = outputs.last_hidden_state
        x = last_hidden_state.permute(0, 2, 1)
        conv_outputs = [torch.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        out = self.fc(x)
        return out.squeeze(1)


#  定义早停机制
class EarlyStopping:
    def __init__(self, patience=3, delta=1e-4):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, path, val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f" EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, path, val_loss)
            self.counter = 0

    def save_checkpoint(self, model, path, val_loss):
        torch.save(model.state_dict(), path)
        print(f" Model saved with improved Val MSE: {val_loss:.4f}")


# 定义文本情感分析器
class TextExtractor:
    def __init__(self, pretrain_model=None, details=True, **kwargs):
        self.details = details
        self.DEVICE = kwargs.get(
            "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.MAX_LEN = kwargs.get("max_len", 64)  # 文本截断长度
        self.EPOCHS = kwargs.get("epochs", 50)
        self.LR = kwargs.get("lr", 2e-6)
        self.MODEL_NAME = kwargs.get("model_name", "bert-base-chinese")
        self.SAVE_PATH = kwargs.get("save_path", "saved_model/best_text_model.pth")
        os.makedirs(os.path.dirname(self.SAVE_PATH), exist_ok=True)

        if details:
            print(f"Device: {self.DEVICE}, Max Length: {self.MAX_LEN}")
            print(f"Model Name: {self.MODEL_NAME}")

        # 标签配置
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
        ]  # 离散化标签
        self.model = BertTextCNNRegressor(self.MODEL_NAME).to(self.DEVICE)

        if pretrain_model is not None:
            self.model.load_state_dict(
                torch.load(pretrain_model, map_location=self.DEVICE)
            )
            if self.details:
                print(f"Loaded pretrained model from {pretrain_model}")
        self.optimizer = AdamW(self.model.parameters(), lr=self.LR, weight_decay=1e-4)
        self.criterion = nn.MSELoss()

    # 将连续的情感值离散化到规定的值
    def discretize(self, value):
        values = np.array(self.DISCRETE_VALUES)
        return float(values[np.argmin(np.abs(values - value))])

    # 单条预测接口
    def predict(self, text, **kwargs):
        discretized = kwargs.get("discretized", False)
        self.model.eval()
        processor = MakeProcessor(mode="text", model_name=self.MODEL_NAME)
        inputs = processor.process(text, max_len=self.MAX_LEN)
        inputs = {k: v.to(self.DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return self.discretize(outputs.item()) if discretized else outputs.item()

    def train(self, train_loader, val_loader):
        if self.details:
            print(f"Training for {self.EPOCHS} epochs with learning rate {self.LR}")
        save_rule = EarlyStopping(patience=8, delta=1e-4)
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
                train_loader, desc=f"Epoch {epoch+1}/{self.EPOCHS}", leave=False
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
                progress_bar.set_postfix(loss=loss.item())
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{self.EPOCHS} - Avg Loss: {avg_loss:.4f}")
            val_loss = self.evaluate(val_loader)
            save_rule(val_loss, self.model, self.SAVE_PATH)

            if save_rule.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        progress_bar = tqdm(val_loader, desc="Validating", leave=False)

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
