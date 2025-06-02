import os
import torch
import whisper
import config
from moviepy.editor import VideoFileClip
class TextProcessor:
    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is None:
            print(f"加载 Whisper 模型：{config.WHISPER_MODEL_NAME} ...")
            self._model = whisper.load_model(config.WHISPER_MODEL_NAME)
        return self._model
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
    def transcribe(self, video_path: str, save_folder: str = None) -> str:
        """
        使用 Whisper 将视频文件转成文字，并保存为 .txt。
        Args:
            video_path: 输入视频文件路径
            save_folder: 转写后文本保存目录，如果为 None，则使用 config.TRANSCRIPT_FOLDER
        Returns:
            text: 返回转写后的文本内容
        """
        os.makedirs(save_folder, exist_ok=True)
        audio_path = self.extract_audio_from_video(video_path)
        model = self._load_model()
        result = model.transcribe(audio_path, fp16=torch.cuda.is_available())
        text = result["text"]
        os.remove(audio_path)  # 删除临时音频文件
        return text
