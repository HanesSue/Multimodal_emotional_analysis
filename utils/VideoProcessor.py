import cv2
import torch
from torchvision import transforms
from typing import List
class VideoProcessor:
    def __init__(self, frame_interval: int = 10):
        self.frame_interval = frame_interval
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def process(self, split_path: str) -> List[torch.Tensor]:
        """
        找到第 idx 条视频对应的 (frames_list, target_score)
        frames_list: List[Tensor], shape (3,224,224) 每个元素是一帧经过 transform 后的张量
        target_score: float, 从 meta_df 对应的列读取
        """
        cap = cv2.VideoCapture(split_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {split_path}")

        frames_tensors: List[torch.Tensor] = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 逐帧抽样
        for frame_idx in range(0, frame_count, self.frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = self.transform(frame_rgb)  # 形状 (3,224,224)
            frames_tensors.append(tensor)

        cap.release()
        # 如果一段视频抽不到任何帧，则 frames_tensors 可能为空——这里简单跳过：
        if len(frames_tensors) == 0:
            # 为了不破坏 batch，大多数做法是至少返回一帧全零张量
            fake = torch.zeros(3, 224, 224)
            frames_tensors = [fake]
        return frames_tensors
    def process_single_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """
        处理单帧图像
        :param frame: 输入的单帧图像张量
        :return: 处理后的张量
        """
        return [self.transform(frame)]