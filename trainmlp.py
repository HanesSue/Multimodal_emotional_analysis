import torch
from models.FusionMLP import FusionMLP
from utils.SIMSData import SIMSLoader
from tqdm import tqdm
import numpy as np
import pandas as pd


if __name__ == "__main__":
    """dataloader = SIMSLoader(
        root="./data/ch-sims2s/ch-simsv2s", mode="mlp", batch_size=64, num_workers=4
    )

    train_loader = dataloader.trainloader
    test_loader = dataloader.testloader"""
    data = pd.read_csv("data\\single_results\\Results.csv")
    label_A = data["sentiment_A"]
    label_T = data["sentiment_T"]
    label_V = data["sentiment_V"]
    label = data["label"]
    X = torch.tensor(np.stack([label_A, label_T, label_V], axis=1), dtype=torch.float32)
    y = torch.tensor(label, dtype=torch.float32)
    print(X.shape)
    data = torch.utils.data.TensorDataset(X, y)
    train_loader = torch.utils.data.DataLoader(
        data, batch_size=64, shuffle=True, num_workers=0
    )
    epoch = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionMLP().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for i in range(epoch):
        model.train()
        for data in tqdm(train_loader):
            optimizer.zero_grad()
            x, y = data
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {i} loss: {loss.item()}")

        torch.save(model.state_dict(), "./checkpoints/mlp_with_predicts.pth")
