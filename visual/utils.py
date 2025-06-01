# utils.py
import cv2
import os
import numpy as np
from facenet_pytorch import MTCNN
from torchvision import transforms
from PIL import Image
from openface import AUDetector  # 假设使用类似 openface 的工具包

mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_faces(video_path):
    cap = cv2.VideoCapture(video_path)
    faces = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        face_boxes, _ = mtcnn.detect(img)
        if face_boxes is not None:
            for box in face_boxes:
                x1, y1, x2, y2 = map(int, box)
                face = img.crop((x1, y1, x2, y2))
                faces.append(transform(face))
    cap.release()
    return faces

def extract_au_features(faces):
    au_model = AUDetector()  # 假设有一个封装的 AU 检测类
    au_features = []
    for face in faces:
        au = au_model.predict(face.permute(1, 2, 0).numpy())  # 输入为 HWC 格式
        au_features.append(torch.tensor(au, dtype=torch.float))
    return au_features