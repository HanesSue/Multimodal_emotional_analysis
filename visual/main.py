# main.py
import torch
from torch.utils.data import DataLoader
from model import EmotionModel
from utils import extract_faces, extract_au_features

# 假设定义好了 dataset 和 dataloader
# from dataset import EmotionVideoDataset

model = EmotionModel()
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for batch in train_loader:
        faces, aus, labels = batch  # [B, T, C, H, W], [B, T, AU], [B]
        faces, aus, labels = faces.cuda(), aus.cuda(), labels.cuda()

        logits = model(faces, aus)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), f'checkpoint_epoch_{epoch}.pth')
