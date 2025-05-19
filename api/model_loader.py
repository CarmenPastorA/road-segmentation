# api/model_loader.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import logging
from models.unet_variants import get_unet_variant

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path="models/best_attention.pth"):
    """
    Loads the production model (AttU_Net) with trained weights.

    Args:
        model_path (str): Path to the .pth file with trained weights.

    Returns:
        model (torch.nn.Module): Loaded model in evaluation mode.
        device (torch.device): Device used (CPU or CUDA).
    """
    logger.info(f"Loading model from {model_path}")
    if not os.path.exists(model_path):
        logger.error(f"Model file {model_path} not found")
        raise FileNotFoundError(f"Model file {model_path} not found")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_unet_variant("attunet", in_channels=3, out_channels=1).to(device)

    # Load weights with weights_only=True for security
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        logger.info(f"Model loaded successfully on {device}")
    except Exception as e:
        logger.error(f"Failed to load model weights: {str(e)}")
        raise

    model.eval()
    return model, device
