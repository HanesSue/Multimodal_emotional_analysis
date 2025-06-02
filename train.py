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

from dataset import MultimodalEmotionDataset, multimodal_collate_fn
from models.models import VideoExtractor, TextExtractor, AudioExtractor, FusionMLP
import config
# 如果使用 Whisper tokenizer，需要在 data_processing 中已经准备好文本字符串
# 这里假设你有一个简单的文本 encoder（如 BERT / RoBERTa / LSTM 等），可以自行替换
# 本示例暂不包含具体 Tokenizer 代码，只给出示意

# ---------------------------------------------------
# 1. 初始化 Dataset & DataLoader
# ---------------------------------------------------
train_dataset = MultimodalEmotionDataset(mode="train") 
train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    collate_fn=multimodal_collate_fn,
    pin_memory=True,
)
test_dataset = MultimodalEmotionDataset(mode="test")  # 如果有测试集，可以类似初始化
test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    collate_fn=multimodal_collate_fn,
    pin_memory=True,
)
val_dataset = MultimodalEmotionDataset(mode="valid")  # 验证集
val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    collate_fn=multimodal_collate_fn,
    pin_memory=True,
)

# ---------------------------------------------------
# 2. 加载预训练编码器 & 融合 MLP
# ---------------------------------------------------
# 视觉编码器
visual_encoder = PretrainedEncoderWrapper(model_path=config.VISUAL_ENCODER_PATH).to(DEVICE)
# 音频编码器
audio_encoder = PretrainedEncoderWrapper(model_path=config.AUDIO_ENCODER_PATH).to(DEVICE)
# 文本编码器
text_encoder = PretrainedEncoderWrapper(model_path=config.TEXT_ENCODER_PATH).to(DEVICE)

# 融合 MLP
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

    for batch in train_pbar:
        # ---------------------------
        # 4.1 读取数据
        # ---------------------------
        video_paths = batch["video_paths"]
        audio_paths = batch["audio_paths"]
        texts = batch["texts"]
        labels = batch["labels"].to(DEVICE)  # (batch_size,)

        # ---------------------------
        # 4.2 视觉特征抽取
        # ---------------------------
        # 示例：假设你的视觉编码器输入需要的是 (B, C, T, H, W) 形式的张量
        # 这里我们用伪代码表示：
        #   v_inputs = []
        #   for vp in video_paths:
        #       # ① 用 cv2 或 torchvision.io 读取视频帧序列
        #       # ② 抽取每隔 N 帧/或首尾帧
        #       # ③ resize & normalize
        #       #   frame_tensor: (C, T, H, W)
        #       v_inputs.append(frame_tensor)
        #   v_inputs = torch.stack(v_inputs, dim=0).to(DEVICE)
        #
        #   v_feats = visual_encoder(v_inputs)  # (batch_size, VISUAL_FEATURE_DIM)
        #
        # 你需要根据实际视觉编码器结构补全此处：
        # ----------------------------------------------------------------------------
        # 假设视觉编码器只需要“预提取好”的512维特征张量，这里我们直接随机生成示例张量：
        v_feats = torch.randn(len(video_paths), 512).to(DEVICE)
        v_feats = visual_encoder(v_feats)  # (batch_size, VISUAL_FEATURE_DIM)
        # ----------------------------------------------------------------------------

        # ---------------------------
        # 4.3 音频特征抽取
        # ---------------------------
        # 示例：假设你的音频编码器输入需要的是 (B, 1, L) 形式原始波形 或 (B, mel_bins, time_frames)
        # 这里我们用伪代码表示：
        #   a_inputs = []
        #   for ap in audio_paths:
        #       waveform, sr = torchaudio.load(ap)  # (1, L), 采样率 sr
        #       # ① 如果需要转 mel-spectrogram，可用 torchaudio.transforms.MelSpectrogram
        #       # ② 或者直接将 waveform 输入到音频编码器
        #       a_inputs.append(mel_spec_tensor)
        #   a_inputs = torch.stack(a_inputs, dim=0).to(DEVICE)
        #
        #   a_feats = audio_encoder(a_inputs)  # (batch_size, AUDIO_FEATURE_DIM)
        #
        # 你需要根据实际音频编码器结构补全此处：
        # ----------------------------------------------------------------------------
        a_feats = torch.randn(len(audio_paths), 512).to(DEVICE)
        a_feats = audio_encoder(a_feats)  # (batch_size, AUDIO_FEATURE_DIM)
        # ----------------------------------------------------------------------------

        # ---------------------------
        # 4.4 文本特征抽取
        # ---------------------------
        # 示例：假设你的文本编码器需要的是 token_ids 张量 (B, seq_len)
        # 你可以在此先做简单 tokenize，再送到文本编码器
        #   tokenized_list = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        #   t_inputs = tokenized_list["input_ids"].to(DEVICE)
        #   t_feats = text_encoder(t_inputs)  # (batch_size, TEXT_FEATURE_DIM)
        #
        # 你需要根据实际文本编码器结构补全此处：
        # ----------------------------------------------------------------------------
        t_feats = torch.randn(len(texts), 512).to(DEVICE)
        t_feats = text_encoder(t_feats)  # (batch_size, TEXT_FEATURE_DIM)
        # ----------------------------------------------------------------------------

        # ---------------------------
        # 4.5 融合 & 计算损失
        # ---------------------------
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
        for batch in val_dataloader:
            video_paths = batch["video_paths"]
            audio_paths = batch["audio_paths"]
            texts = batch["texts"]
            labels = batch["labels"].to(DEVICE)

            # 视觉特征
            v_feats = torch.randn(len(video_paths), 512).to(DEVICE)
            v_feats = visual_encoder(v_feats)

            # 音频特征
            a_feats = torch.randn(len(audio_paths), 512).to(DEVICE)
            a_feats = audio_encoder(a_feats)

            # 文本特征
            t_feats = torch.randn(len(texts), 512).to(DEVICE)
            t_feats = text_encoder(t_feats)

            # 融合预测
            preds = fusion_mlp(v_feats, a_feats, t_feats)

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
    for batch in test_dataloader:
        video_paths = batch["video_paths"]
        audio_paths = batch["audio_paths"]
        texts = batch["texts"]
        labels = batch["labels"].to(DEVICE)

        # 视觉特征
        v_feats = torch.randn(len(video_paths), 512).to(DEVICE)
        v_feats = visual_encoder(v_feats)

        # 音频特征
        a_feats = torch.randn(len(audio_paths), 512).to(DEVICE)
        a_feats = audio_encoder(a_feats)

        # 文本特征
        t_feats = torch.randn(len(texts), 512).to(DEVICE)
        t_feats = text_encoder(t_feats)

        # 融合预测
        preds = fusion_mlp(v_feats, a_feats, t_feats)

        loss = criterion(preds, labels)
        test_loss += loss.item() * labels.size(0)
test_avg_loss = test_loss / len(test_dataset)
print(f"测试集平均损失: {test_avg_loss:.4f}")
