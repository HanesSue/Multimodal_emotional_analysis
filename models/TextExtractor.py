# -*- coding : utf-8 -*-
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


# 定义数据集
class TextRegressionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.float)
        return item


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

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
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
    """
    文本情感分析器：
    参数 type:   "train" (训练新的模型)  "load" (加载已经训练好的模型)

            第一种工作方式:   type = "train" (训练新的模型)

            初始化后，调用train()方法进行训练，训练完成后
            调用predict()方法进行预测

            第二种工作方式:   type = "load"  (加载已经训练好的模型)

            初始化后，调用predict()方法进行预测

    参数 details:  是否打印具体信息

    """

    def __init__(self, type="train", details=True, **kwargs):

        self.type = type
        self.details = details
        self.DEVICE = kwargs.get(
            "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.MAX_LEN = kwargs.get("max_len", 64)  # 文本截断长度
        self.BATCH_SIZE = kwargs.get("batch_size", 32)
        self.EPOCHS = kwargs.get("epochs", 50)
        self.LR = kwargs.get("lr", 2e-6)
        self.MODEL_NAME = kwargs.get("model_name", "bert-base-chinese")
        self.SAVE_PATH = kwargs.get("save_path", "saved_model/best_text_model.pth")
        self.data_path = kwargs.get("data_path", "dataset/meta.csv")

        if details:
            print("正在初始化文本情感分析器...")

        # 标签配置
        self.DISCRETE_VALUES = [
            -1,
            -0.8,
            -0.6,
            -0.4,
            -0.2,
            0,
            0.2,
            0.4,
            0.6,
            0.8,
            1,
        ]  # 离散化标签

        if details:
            print("加载数据集...")
        # 加载数据集
        self.df = pd.read_csv(self.data_path)
        self.df = self.df.dropna(subset=["text", "label_T", "mode"])
        self.df["label_T"] = self.df["label_T"].astype(float)

        self.train_df = self.df[self.df["mode"] == "train"]
        self.val_df = self.df[self.df["mode"] == "valid"]

        self.train_texts = self.train_df["text"].tolist()
        self.train_labels = self.train_df["label_T"].tolist()
        self.val_texts = self.val_df["text"].tolist()
        self.val_labels = self.val_df["label_T"].tolist()
        if details:
            print("加载预训练Bert-Chinese模型...")
        # 初始化tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained(self.MODEL_NAME)

        # 初始化数据集和数据加载器
        self.train_dataset = TextRegressionDataset(
            self.train_texts, self.train_labels, self.tokenizer, self.MAX_LEN
        )
        self.val_dataset = TextRegressionDataset(
            self.val_texts, self.val_labels, self.tokenizer, self.MAX_LEN
        )
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.BATCH_SIZE, shuffle=True
        )
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.BATCH_SIZE)

        # 根据type加载模型

        if self.type == "train":
            # 初始化模型、损失函数和优化器
            self.model = BertTextCNNRegressor(self.MODEL_NAME).to(self.DEVICE)
            self.criterion = nn.MSELoss()
            self.optimizer = AdamW(self.model.parameters(), lr=self.LR)
            self.total_steps = len(self.train_loader) * self.EPOCHS
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=int(0.1 * self.total_steps),
                num_training_steps=self.total_steps,
            )
            if details:
                print("模型初始化完毕,可调用train()方法进行训练")

        elif self.type == "load":
            if details:
                print("正在加载模型...")
            self.model = self.load_model()
            if details:
                print("模型加载完毕！")

    # 将连续的情感值离散化到规定的值
    def discretize(self, value):
        values = np.array(self.DISCRETE_VALUES)
        return float(values[np.argmin(np.abs(values - value))])

    # 模型保存函数
    def load_model(self):
        path = self.SAVE_PATH
        model = BertTextCNNRegressor(self.MODEL_NAME).to(self.DEVICE)
        model.load_state_dict(torch.load(path, map_location=self.DEVICE))
        model.eval()
        return model

    # 单条预测接口
    def predict(self, text):
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.MAX_LEN,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.DEVICE)
        attention_mask = encoding["attention_mask"].to(self.DEVICE)
        with torch.no_grad():
            output = self.model(input_ids, attention_mask)
        raw_pred = output.item()
        discrete_pred = self.discretize(raw_pred)
        return discrete_pred

    """
    # ===================== 示例测试 =====================
        sample_texts = [
            "这是一条测试文本，用于预测连续标签",
            "这朵花儿也太美了吧！",
            "我讨厌这个角色"
        ]

        for text in sample_texts:
            pred = predict(text)
            print(f"文本: {text}\n离散化预测值: {pred}\n")
    """

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        progress = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in progress:
            input_ids = batch["input_ids"].to(self.DEVICE)
            attention_mask = batch["attention_mask"].to(self.DEVICE)
            labels = batch["labels"].to(self.DEVICE)

            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())
        return total_loss / len(self.train_loader)

    def eval_epoch(self):
        self.model.eval()
        preds, trues = [], []
        progress = tqdm(self.val_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for batch in progress:
                input_ids = batch["input_ids"].to(self.DEVICE)
                attention_mask = batch["attention_mask"].to(self.DEVICE)
                labels = batch["labels"].to(self.DEVICE)

                outputs = self.model(input_ids, attention_mask)
                preds.extend(outputs.cpu().numpy())
                trues.extend(labels.cpu().numpy())
        mse = mean_squared_error(trues, preds)
        return mse

    def train(self):
        print("\n 开始训练...")
        early_stopper = EarlyStopping(patience=8, delta=1e-4)

        for epoch in range(self.EPOCHS):
            print(f"\n Epoch {epoch+1}/{self.EPOCHS}")
            train_loss = self.train_epoch()
            val_loss = self.eval_epoch()
            print(f" Train Loss: {train_loss:.4f} | Val MSE: {val_loss:.4f}")

            early_stopper(val_loss, self.model, self.SAVE_PATH)

            if early_stopper.early_stop:
                print(f" Early stopping triggered at epoch {epoch+1}")
                break

        print("\n 训练完成！")

    def predict_dataset(self, dataloader):
        self.model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting on Train Set"):
                input_ids = batch["input_ids"].to(self.DEVICE)
                attention_mask = batch["attention_mask"].to(self.DEVICE)
                labels = batch["labels"].cpu().numpy()

                outputs = self.model(input_ids, attention_mask)
                predictions = outputs.cpu().numpy()

                preds.extend(predictions)
                trues.extend(labels)
        return trues, preds

    def predict_train(self):
        # 获取训练集的真实值与预测值
        train_trues, train_preds = self.predict_dataset(self.train_loader)
        # 打印前20个样本的真实值与预测值对比
        print("\n=== 训练集真实值 vs 预测值（前20条） ===")
        for i in range(min(20, len(train_trues))):
            raw_pred = train_preds[i]
            raw_true = train_trues[i]
            print(
                f"真实值: {raw_true:.1f} | 预测值: {raw_pred:.4f} | 离散化: {self.discretize(raw_pred):.1f}"
            )
    def predict_batch(self, text_batch):
        """
        text_batch为文本的列表，长度为n

        输出为Tensor(维度为 n * 1 )
        """
        
        # 对每个文本调用self.predict方法并收集结果
        predictions = [self.predict(text) for text in text_batch]
        
        # 将结果列表转换为张量并调整形状为 n*1
        return torch.tensor(predictions).reshape(-1, 1)
