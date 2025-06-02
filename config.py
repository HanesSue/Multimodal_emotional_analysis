# config.py

import os

####################################
#       路径 & 文件配置
####################################
# 原始视频文件目录
VIDEO_FOLDER = "./data/ch-sims2s/ch-simsv2s/Raw"  # 例如: "./data/videos"
# 提取后的音频保存目录
AUDIO_FOLDER = "./data/audios"
# Whisper 转写后的文本保存目录（可选：也可以直接在内存中使用，不存文件）
TRANSCRIPT_FOLDER = "./data/transcripts"

# 三个预训练编码器的权重文件路径
# 假设分别是：visual_encoder.pt, audio_encoder.pt, text_encoder.pt
VISUAL_ENCODER_PATH = "./pretrained/visual_encoder.pt"
AUDIO_ENCODER_PATH = "./pretrained/audio_encoder.pt"
TEXT_ENCODER_PATH = "./pretrained/text_encoder.pt"

# 标签文件：CSV 格式，包含两列 [video_filename, label]
# 例如：
# video1.mp4,0
# video2.mp4,1
LABEL_CSV = "./data/ch-sims2s/ch-simsv2s/meta.csv"

# 训练时保存模型的目录
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

####################################
#       模型 & 训练超参数
####################################
# MLP 隐藏层维度
MLP_HIDDEN_DIM = 256
# 三个编码器各自输出的特征维度（需要与预训练时一致）
VISUAL_FEATURE_DIM = 128
AUDIO_FEATURE_DIM = 128
TEXT_FEATURE_DIM = 128

# 学习率、批大小、训练轮数等
LR = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 10

# 是否在训练时微调预训练编码器（False 表示编码器 freeze，仅训练 MLP）
FINETUNE_ENCODERS = False

####################################
#       Whisper (ASR) 配置
####################################
# Whisper 模型名，可选 "tiny", "base", "small", "medium", "large"
WHISPER_MODEL_NAME = "base"

####################################
#       设备配置
####################################
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
