# train.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    NUM_EPOCHS,
    LR,
    DEVICE,
    FINETUNE_ENCODERS,
    CHECKPOINT_DIR,
)

from utils import SIMSData 
from models import VideoExtractor, TextExtractor, AudioExtractor, FusionMLP
import config
from utils.SIMSData import SIMSData 
# 如果使用 Whisper tokenizer，需要在 data_processing 中已经准备好文本字符串
# 这里假设你有一个简单的文本 encoder（如 BERT / RoBERTa / LSTM 等），可以自行替换
# 本示例暂不包含具体 Tokenizer 代码，只给出示意

# ---------------------------------------------------
# 1. 初始化 Dataset & DataLoader
# ---------------------------------------------------
train_dataset = SIMSData(mode="train",root=config.DATA_ROOT)  # 替换为你的数据集路径
train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
test_dataset = SIMSData(mode="test",root=config.DATA_ROOT)  # 测试集，可以类似初始化
test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)
val_dataset = SIMSData(mode="valid",root=config.DATA_ROOT)  # 验证集
val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

# ---------------------------------------------------
# 2. 加载预训练编码器 & 融合 MLP
# ---------------------------------------------------
visual_encoder = VideoExtractor(type='load', datails=True, save_path=config.VISUAL_ENCODER_PATH, data_path=config.LABEL_CSV).to(DEVICE)
audio_encoder = AudioExtractor(model_path=config.AUDIO_ENCODER_PATH).to(DEVICE)
text_encoder = TextExtractor(type='load', datails=True, save_path=config.VISUAL_ENCODER_PATH, data_path=config.LABEL_CSV).to(DEVICE)

fusion_mlp = FusionMLP().to(DEVICE)

# ---------------------------------------------------
# 3. 定义优化器 & 损失函数
# ---------------------------------------------------
# 收集需要优化的参数
params = list(fusion_mlp.parameters())
if FINETUNE_ENCODERS:
    params += list(visual_encoder.parameters())
    params += list(audio_encoder.parameters())
    params += list(text_encoder.parameters())

optimizer = torch.optim.Adam(params, lr=LR)
criterion = nn.MSELoss()  # 假设标签是连续值回归；如果分类，可换成 CrossEntropyLoss
best_val_loss = float("inf")  # 用于保存最优模型
# 确保检查点目录存在
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
# ---------------------------------------------------
# 4. 训练循环
# ---------------------------------------------------
for epoch in range(NUM_EPOCHS):
    visual_encoder.train() if FINETUNE_ENCODERS else visual_encoder.eval()
    audio_encoder.train() if FINETUNE_ENCODERS else audio_encoder.eval()
    text_encoder.train() if FINETUNE_ENCODERS else text_encoder.eval()
    fusion_mlp.train()

    epoch_loss = 0.0
    train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

    for video_tensor,audio_tensor,texts,labels in train_pbar:
        video_tensor = video_tensor.to(DEVICE)
        audio_tensor = audio_tensor.to(DEVICE)
        texts = texts
        labels = labels.to(DEVICE)  # (batch_size,)

        v_feats = visual_encoder(video_tensor)  # (batch_size, VISUAL_FEATURE_DIM)
        a_feats = audio_encoder(audio_tensor)  # (batch_size, AUDIO_FEATURE_DIM)
        t_feats = text_encoder(texts)  # (batch_size, TEXT_FEATURE_DIM)

        preds = fusion_mlp(v_feats, a_feats, t_feats)  # (batch_size,)

        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * labels.size(0)
        train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    avg_loss = epoch_loss / len(train_dataset)
    print(f"Epoch {epoch+1} 完成，平均 Loss: {avg_loss:.4f}")

    visual_encoder.eval()
    audio_encoder.eval()
    text_encoder.eval()
    fusion_mlp.eval()
    
    val_epoch_loss = 0.0
    with torch.no_grad():
        for video_tensor,audio_tensor,texts,labels in val_dataloader:
            video_tensor = video_tensor.to(DEVICE)
            audio_tensor = audio_tensor.to(DEVICE)
            texts = texts
            labels = labels.to(DEVICE)  # (batch_size,)

            v_feats = visual_encoder(video_tensor)  # (batch_size, VISUAL_FEATURE_DIM)
            a_feats = audio_encoder(audio_tensor)  # (batch_size, AUDIO_FEATURE_DIM)
            t_feats = text_encoder(texts)  # (batch_size, TEXT_FEATURE_DIM)

            preds = fusion_mlp(v_feats, a_feats, t_feats)  # (batch_size,)

            loss = criterion(preds, labels)
            val_epoch_loss += loss.item() * labels.size(0)
    
    val_avg_loss = val_epoch_loss / len(val_dataset)
    # ---------------------------
    # 4.6 保存 Checkpoint
    # ---------------------------
    if val_avg_loss < best_val_loss:
        best_val_loss = val_avg_loss
        print(f"验证集损失降低到 {val_avg_loss:.4f}，保存模型...")
        ckpt = {
            "fusion_mlp_state_dict": fusion_mlp.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,
        }
        if FINETUNE_ENCODERS:
            ckpt["visual_encoder_state_dict"] = visual_encoder.state_dict()
            ckpt["audio_encoder_state_dict"] = audio_encoder.state_dict()
            ckpt["text_encoder_state_dict"] = text_encoder.state_dict()

        save_path = os.path.join(CHECKPOINT_DIR, f"multimodal_epoch{epoch+1}.pth")
        torch.save(ckpt, save_path)
        print(f"已保存模型到 {save_path}\n")

print("训练完成。")
# ---------------------------------------------------
# 5. 测试集评估
# ---------------------------------------------------
visual_encoder.eval()
audio_encoder.eval()
text_encoder.eval()
fusion_mlp.eval()
test_loss = 0.0
with torch.no_grad():
    for video_tensor,audio_tensor,texts,labels in test_dataloader:
        video_tensor = video_tensor.to(DEVICE)
        audio_tensor = audio_tensor.to(DEVICE)
        texts = texts
        labels = labels.to(DEVICE)  # (batch_size,)

        v_feats = visual_encoder(video_tensor)  # (batch_size, VISUAL_FEATURE_DIM)
        a_feats = audio_encoder(audio_tensor)  # (batch_size, AUDIO_FEATURE_DIM)
        t_feats = text_encoder(texts)  # (batch_size, TEXT_FEATURE_DIM)

        preds = fusion_mlp(v_feats, a_feats, t_feats)  # (batch_size,)

        loss = criterion(preds, labels)
        test_loss += loss.item() * labels.size(0)
test_avg_loss = test_loss / len(test_dataset)
print(f"测试集平均损失: {test_avg_loss:.4f}")
