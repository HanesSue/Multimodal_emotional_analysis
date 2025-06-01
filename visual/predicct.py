# predict.py
import torch
from model import EmotionModel
from utils import extract_faces, extract_au_features
from voting import vote_average

video_path = "test_video.mp4"
faces = extract_faces(video_path)
au_feats = extract_au_features(faces)

model = EmotionModel()
model.load_state_dict(torch.load("checkpoint_epoch_9.pth"))
model.eval()
model = model.cuda()

# 按帧级分批预测情绪
predictions = []
with torch.no_grad():
    for i in range(len(faces)):
        face = faces[i].unsqueeze(0).unsqueeze(0).cuda()  # [1, 1, C, H, W]
        au = au_feats[i].unsqueeze(0).unsqueeze(0).cuda()  # [1, 1, D]
        pred = model(face, au)[0]
        predictions.append(pred.cpu())

final_label = vote_average(predictions)
print("视频整体预测情绪标签：", final_label)
