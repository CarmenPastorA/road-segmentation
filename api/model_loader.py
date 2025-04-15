import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from models.mini_unet import MiniUNet  # or baseline_unet.MiniUNet, etc.

def load_model(model_path="models/best_mini_unet.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniUNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device
