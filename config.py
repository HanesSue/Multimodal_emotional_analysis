import cv2
from mtcnn import MTCNN
import torch
from torchvision import transforms
from models import VisualExtractor  # 替换为你自己的路径和类名
import numpy as np

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VisualExtractor().to(device)
model.load_state_dict(torch.load('visual_encoder.pt', map_location=device))
model.eval()

# 图像预处理方法（你训练时用的transform）
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 初始化 MTCNN
detector = MTCNN()

# 视频路径
input_path = '004_VID.mp4'
output_path = 'output_with_emotion.mp4'

cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    print(f"处理帧 {frame_idx}")
    frame_idx += 1

    # 检测人脸
    results = detector.detect_faces(frame)

    # 整帧送入模型预测情绪（也可以替换为送入人脸区域）
    input_tensor = transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor)

    # 显示每个人脸框，并在上方写入情绪标签
    for res in results:
        x, y, w, h = [int(v) for v in res['box']]
        x, y = max(0, x), max(0, y)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"情绪值：{logits}", (x, max(0, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
