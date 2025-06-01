# data_processing.py

import os
import whisper
import torch
from moviepy.editor import VideoFileClip
from tqdm import tqdm

import config

# 确保输出文件夹存在
os.makedirs(config.AUDIO_FOLDER, exist_ok=True)
os.makedirs(config.TRANSCRIPT_FOLDER, exist_ok=True)

# ----------------------------------
# 1. 视频提取音频
# ----------------------------------
def extract_audio_from_video(video_path: str, save_folder: str = None) -> str:
    """
    从视频中提取音频，并保存为 .wav 格式。
    Args:
        video_path: 输入视频文件路径（mp4, avi 等）
        save_folder: 音频文件保存目录，如果为 None，则使用 config.AUDIO_FOLDER
    Returns:
        audio_path: 提取后音频文件路径（wav）
    """
    if save_folder is None:
        save_folder = config.AUDIO_FOLDER

    video_name = os.path.basename(video_path)
    audio_name = os.path.splitext(video_name)[0] + ".wav"
    audio_path = os.path.join(save_folder, audio_name)

    if os.path.exists(audio_path):
        # 如果已经存在，就不重复提取
        return audio_path

    # 使用 moviepy 提取音频
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
    clip.reader.close()
    clip.audio.reader.close_proc()

    return audio_path


# ----------------------------------
# 2. 音频转写 (Whisper)
# ----------------------------------
_whisper_model = None

def _load_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        print(f"加载 Whisper 模型：{config.WHISPER_MODEL_NAME} ...")
        _whisper_model = whisper.load_model(config.WHISPER_MODEL_NAME)
    return _whisper_model

def transcribe_audio(audio_path: str, save_folder: str = None) -> str:
    """
    使用 Whisper 将音频文件转成文字，并保存为 .txt。
    Args:
        audio_path: 输入音频文件路径 (.wav)
        save_folder: 转写后文本保存目录，如果为 None，则使用 config.TRANSCRIPT_FOLDER
    Returns:
        transcript_path: 返回转写后文本文件路径
    """
    if save_folder is None:
        save_folder = config.TRANSCRIPT_FOLDER

    audio_name = os.path.basename(audio_path)
    txt_name = os.path.splitext(audio_name)[0] + ".txt"
    transcript_path = os.path.join(save_folder, txt_name)

    if os.path.exists(transcript_path):
        # 如果已经转写过，直接返回
        return transcript_path

    model = _load_whisper_model()
    # Whisper 直接接受路径输入
    result = model.transcribe(audio_path, fp16=torch.cuda.is_available())
    text = result["text"]

    # 将文字写入文件
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(text)

    return transcript_path


# ----------------------------------
# 3. 读取转写后的文本
# ----------------------------------
def load_transcript(transcript_path: str) -> str:
    """
    读取之前转写保存的文本文件，返回字符串
    """
    with open(transcript_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    return content


# ----------------------------------
# 4. 遍历文件夹下所有视频，一次性批量完成 音频提取 + 转写（可选）
# ----------------------------------
def preprocess_all_videos(video_folder: str = None):
    """
    遍历 video_folder，提取所有音频并转写为文本。
    主要适用于一次性预处理所有数据，减少训练时的 I/O 开销。
    """
    if video_folder is None:
        video_folder = config.VIDEO_FOLDER

    video_list = []
    for ext in [".mp4", ".avi", ".mov", ".mkv"]:
        video_list.extend(
            [os.path.join(video_folder, fn) for fn in os.listdir(video_folder) if fn.endswith(ext)]
        )

    print(f"发现 {len(video_list)} 个视频，开始批量预处理音频和转写 ...")
    for video_path in tqdm(video_list):
        audio_path = extract_audio_from_video(video_path)
        _ = transcribe_audio(audio_path)

    print("预处理完成。")
