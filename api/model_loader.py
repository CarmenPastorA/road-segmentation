import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from models.unet_variants import get_unet_variant


def load_model(model_path="models/best_attention.pth"):
    """
    Loads the production model (AttU_Net) with trained weights.

    Args:
        model_path (str): Path to the .pth file with trained weights.

    Returns:
        model (torch.nn.Module): Loaded model in evaluation mode.
        device (torch.device): Device used (CPU or CUDA).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the AttU_Net using the model factory
    model = get_unet_variant("attunet", in_channels=3, out_channels=1).to(device)

    # Load pretrained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, device
