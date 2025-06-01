# multimodal_emotion_inference.py
# 单一视频、图像、音频、文本的推理接口
import os
import torch
import torchaudio
from moviepy.editor import VideoFileClip
from transformers import pipeline
from models.video_encoder import VideoEncoder
from models.audio_encoder import AudioEncoder
from models.text_encoder import TextEncoder
from models.mlp_fusion import FusionMLP
from utils.video_utils import extract_video_frames
from utils.audio_utils import extract_audio_from_video, preprocess_audio
from utils.text_utils import speech_to_text, preprocess_text


def load_models(device):
    video_encoder = VideoEncoder().to(device)
    audio_encoder = AudioEncoder().to(device)
    text_encoder = TextEncoder().to(device)
    fusion_mlp = FusionMLP().to(device)

    video_encoder.load_state_dict(torch.load('checkpoints/video_encoder.pt', map_location=device))
    audio_encoder.load_state_dict(torch.load('checkpoints/audio_encoder.pt', map_location=device))
    text_encoder.load_state_dict(torch.load('checkpoints/text_encoder.pt', map_location=device))
    fusion_mlp.load_state_dict(torch.load('checkpoints/fusion_mlp.pt', map_location=device))

    video_encoder.eval()
    audio_encoder.eval()
    text_encoder.eval()
    fusion_mlp.eval()

    return video_encoder, audio_encoder, text_encoder, fusion_mlp


def infer_single_video(video_path, device='cuda'):
    video_encoder, audio_encoder, text_encoder, fusion_mlp = load_models(device)

    # 视频帧提取并编码
    video_tensor = extract_video_frames(video_path).to(device)
    with torch.no_grad():
        video_feat = video_encoder(video_tensor)

    # 音频提取并编码
    audio_path = extract_audio_from_video(video_path)
    waveform, sample_rate = preprocess_audio(audio_path)
    with torch.no_grad():
        audio_feat = audio_encoder(waveform.unsqueeze(0).to(device))

    # 文本提取并编码
    text = speech_to_text(audio_path)
    tokens = preprocess_text(text)
    with torch.no_grad():
        text_feat = text_encoder(tokens.to(device))

    # 融合预测
    with torch.no_grad():
        fused_feat = torch.cat([video_feat, audio_feat, text_feat], dim=1)
        emotion_score = fusion_mlp(fused_feat)

    return emotion_score.item()


def infer_single_image(image_path, device='cuda'):
    video_encoder, _, _, fusion_mlp = load_models(device)
    from torchvision import transforms
    from PIL import Image

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        video_feat = video_encoder(image_tensor)
        fused_feat = torch.cat([video_feat, torch.zeros_like(video_feat), torch.zeros_like(video_feat)], dim=1)
        emotion_score = fusion_mlp(fused_feat)
    return emotion_score.item()


def infer_single_audio(audio_path, device='cuda'):
    _, audio_encoder, _, fusion_mlp = load_models(device)
    waveform, sample_rate = preprocess_audio(audio_path)
    with torch.no_grad():
        audio_feat = audio_encoder(waveform.unsqueeze(0).to(device))
        fused_feat = torch.cat([torch.zeros_like(audio_feat), audio_feat, torch.zeros_like(audio_feat)], dim=1)
        emotion_score = fusion_mlp(fused_feat)
    return emotion_score.item()


def infer_single_text(text, device='cuda'):
    with open(text, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    _, _, text_encoder, fusion_mlp = load_models(device)
    tokens = preprocess_text(text)
    with torch.no_grad():
        text_feat = text_encoder(tokens.to(device))
        fused_feat = torch.cat([torch.zeros_like(text_feat), torch.zeros_like(text_feat), text_feat], dim=1)
        emotion_score = fusion_mlp(fused_feat)
    return emotion_score.item()


if __name__ == '__main__':
    video_path = 'demo/demo_video.mp4'
    image_path = 'demo/demo_image.jpg'
    audio_path = 'demo/demo_audio.wav'
    text_input = 'demo/demo_text.txt'

    print('视频情绪分数:', infer_single_video(video_path))
    print('图片情绪分数:', infer_single_image(image_path))
    print('音频情绪分数:', infer_single_audio(audio_path))
    print('文本情绪分数:', infer_single_text(text_input))
