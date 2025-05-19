import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor
from models.mini_unet import MiniUNet
from data.transforms import get_transforms


def predict_single_image(image_path, model_path, device, image_size=256, threshold=0.5):
    """
    Loads an image, applies preprocessing, runs inference using a trained model,
    and returns the original image and predicted binary mask.
    """
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = get_transforms(image_size=image_size)
    transformed = transform(image=image)
    input_tensor = transformed['image'].unsqueeze(0).to(device)

    # Load trained model
    model = MiniUNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = (output.squeeze().cpu().numpy() > threshold).astype('uint8')  # binary mask

    return image, pred_mask


def plot_prediction(image, mask):
    """
    Displays the input image and the predicted segmentation mask side by side.
    """
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Replace with the path to any image you want to test
    image_path = "data/raw/valid/64_sat.jpg"
    model_path = "models/best_mini_unet.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image, mask = predict_single_image(image_path, model_path, device)
    plot_prediction(image, mask)

