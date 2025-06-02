import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils import extract_faces_from_video
import os

class VideoFaceDataset(Dataset):
    def __init__(self, video_paths, labels, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        faces = extract_faces_from_video(self.video_paths[idx])
        if len(faces) == 0:
            faces = [Image.new('RGB', (224, 224))] * 10  # fallback

        face_tensors = [self.transform(face) for face in faces]
        video_tensor = torch.stack(face_tensors)  # [T, C, H, W]
        label = self.labels[idx]
        return video_tensor, torch.tensor(label)
