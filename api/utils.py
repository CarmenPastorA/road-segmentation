#api/utils.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shared.transforms import get_transforms
import numpy as np

def preprocess_image(pil_img, device, image_size=256):
    """
    Preprocesses a PIL image for model inference.
    Returns a torch tensor of shape (1, 3, H, W).
    """
    image = np.array(pil_img)
    transform = get_transforms(image_size=image_size)
    input_tensor = transform(image=image)['image'].unsqueeze(0).to(device)
    return input_tensor

def postprocess_mask(output, threshold=0.5):
    mask = (output.squeeze().detach().cpu().numpy() > threshold).astype('uint8')
    return mask
