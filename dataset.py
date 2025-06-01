# dataset.py

import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from data_processing import extract_audio_from_video, transcribe_audio, load_transcript
from config import VIDEO_FOLDER, LABEL_CSV

class MultimodalEmotionDataset(Dataset):
    """
    自定义多模态情感数据集
    每个样本对应一个视频文件：
      - 视觉：视频路径（后续在 collate_fn 或 DataLoader 内部处理为张量）
      - 音频：首先提取音频文件路径，再做特征
      - 文本：将音频转写文件读取成字符串
      - 标签：从 CSV 中读取
    """

    def __init__(self, video_folder: str = None, label_csv: str = None):
        super(MultimodalEmotionDataset, self).__init__()
        if video_folder is None:
            video_folder = VIDEO_FOLDER
        if label_csv is None:
            label_csv = LABEL_CSV

        self.video_folder = video_folder

        # 读取标签 CSV，假设列名：['video_filename', 'label']
        df = pd.read_csv(label_csv)
        # 保证索引从 0 开始
        self.samples = []
        for idx, row in df.iterrows():
            vid_name = row["video_id"]+"/"+row["clip_id"]+".mp4"  # 视频文件名由 video_id 和 clip_id 组成
            label = float(row["label"])  # 假设情感标签是连续标量；如果是分类，可转为 int
            video_path = os.path.join(self.video_folder, vid_name)
            text=row["text"]  # CSV 中有转写文本列
            if os.path.exists(video_path):
                self.samples.append((video_path, label, text))
            else:
                print(f"WARNING: 视频文件不存在：{video_path}，已跳过。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            dict {
              "video_path": str,
              "audio_path": str,
              "text": str,
              "label": float
            }
        """
        video_path, label, text = self.samples[idx]

        # 1) 提取音频（如果已经提取过，则直接返回已存在路径）
        audio_path = extract_audio_from_video(video_path)

        '''# 2) 音频转写并读取文本
        transcript_path = transcribe_audio(audio_path)
        text = load_transcript(transcript_path)'''

        # 3) 返回三模态输入与标签
        return {
            "video_path": video_path,   # 后面在 collate_fn 中做视频帧抽取或送入视觉编码器
            "audio_path": audio_path,   # 后面在 collate_fn 中做音频特征提取或送入音频编码器
            "text": text,               # 后面在 collate_fn 中做文本 token 化或送入文本编码器
            "label": torch.tensor(label, dtype=torch.float32),
        }


def multimodal_collate_fn(batch):
    """
    自定义 collate_fn，将 batch 中的数据整合成张量：
    - 视觉：这里示例直接取路径列表，由训练脚本中再进一步处理；
    - 音频：示例直接取路径列表，后续再用 torchaudio/其他库加载特征；
    - 文本：示例直接取字符串列表，后续再 tokenize；
    - 标签：batch_size 张量
    你可以根据实际情况，将音频文件直接加载成 Mel-Spectrogram，或者将文本直接 tokenize 为 token_ids。
    """
    video_paths = [item["video_path"] for item in batch]
    audio_paths = [item["audio_path"] for item in batch]
    texts = [item["text"] for item in batch]
    labels = torch.stack([item["label"] for item in batch], dim=0)

    return {
        "video_paths": video_paths,
        "audio_paths": audio_paths,
        "texts": texts,
        "labels": labels,
    }
