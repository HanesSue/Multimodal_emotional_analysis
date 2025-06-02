# multimodal_emotion_inference.py
# 单一视频、图像、音频、文本的推理接口
import argparse
import os
import torch
import torchaudio
from moviepy.editor import VideoFileClip
from transformers import pipeline
from models.VideoExtractor import VideoEncoder
from models.AudioExtractor import AudioEncoder
from models.TextExtractor import TextEncoder
from models.FusionMLP import FusionMLP
from utils.VideoProcessor import VideoProcessor
from utils.AudioProcessor import AudioProcessor
from utils.TextProcessor import TextProcessor
import sys
from torchvision import transforms
from PIL import Image

def infer_single_video(video_path, device='cuda'):
    video_encoder= VideoEncoder(type='load', datails=True, save_path='path/to/video_encoder.pth', data_path='path/to/label.csv').to(device)
    audio_encoder = AudioEncoder(model_path='path/to/audio_encoder.pth').to(device)
    text_encoder = TextEncoder(type='load', datails=True, save_path='path/to/text_encoder.pth', data_path='path/to/label.csv').to(device)
    fusion_mlp = FusionMLP(input_dim=1024*3, output_dim=1).to(device)
    video_processor = VideoProcessor()
    audio_processor = AudioProcessor()
    text_processor = TextProcessor()
    # 视频帧提取并编码
    video_tensor = video_processor.process(video_path).to(device)
    audio_tensor = audio_processor.process(video_path).to(device)
    text = text_processor.transcribe(video_path)
    with torch.no_grad():
        video_feat = video_encoder(video_tensor)
        audio_feat = audio_encoder(audio_tensor)
        text_feat = text_encoder(text)
        fused_feat = torch.cat([video_feat, audio_feat, text_feat], dim=1)
        emotion_score = fusion_mlp(fused_feat)

    return emotion_score.item()


def infer_single_image(image_path, device='cuda'):
    video_encoder= VideoEncoder(type='load', datails=True, save_path='path/to/video_encoder.pth', data_path='path/to/label.csv').to(device)
    with torch.no_grad():
        image_tensor = VideoProcessor().process(image_path).to(device)
        emotion_score = video_encoder(image_tensor)
    return emotion_score[0].item()


def infer_single_audio(audio_path, device='cuda'):
    audio_encoder = AudioEncoder(model_path='path/to/audio_encoder.pth').to(device)
    with torch.no_grad():
        audio_tensor = AudioProcessor().process(audio_path).to(device)
        emotion_score = audio_encoder(audio_tensor)
    return emotion_score.item()


def infer_single_text(text, device='cuda'):
    text_encoder = TextEncoder(type='load', datails=True, save_path='path/to/text_encoder.pth', data_path='path/to/label.csv').to(device)
    with torch.no_grad():
        text_tensor = text_encoder(text).to(device)
        emotion_score = text_tensor.mean()  # 假设文本编码器输出的是一个向量，取平均作为情绪分数
    return emotion_score.item()
def infer_by_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.mp4', '.avi', '.mov', '.mkv']:
        print('视频情绪分数:', infer_single_video(file_path))
    elif ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        print('图片情绪分数:', infer_single_image(file_path))
    elif ext in ['.wav', '.mp3', '.flac']:
        print('音频情绪分数:', infer_single_audio(file_path))
    elif ext in ['.txt']:
        print('文本情绪分数:', infer_single_text(file_path))
    else:
        print('不支持的文件类型:', ext)

if __name__ == '__main__':
    Parser= argparse.ArgumentParser(description='多模态情绪推理接口')
    Parser.add_argument('--file_path', type=str, help='文件路径')
    infer_by_file(Parser.parse_args().file_path)
