# fer_plus_model.py
import torch
from torchvision import models
import torch.nn as nn

class FERPlusNet(nn.Module):
    def __init__(self, num_classes=7):
        super(FERPlusNet, self).__init__()
        self.base = models.resnet18(pretrained=True)
        self.base.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.base(x)

def load_ferplus_model(checkpoint_path=None):
    model = FERPlusNet()
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))
    return model