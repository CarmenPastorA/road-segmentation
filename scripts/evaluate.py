import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.mini_unet import MiniUNet
from data.dataset import RoadSegmentationDataset
from data.transforms import get_transforms


def iou_score(preds, targets, threshold=0.5, eps=1e-6):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = ((preds + targets) >= 1).float().sum(dim=(1, 2, 3))
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def evaluate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_iou = 0.0

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            val_loss += loss.item()
            val_iou += iou_score(outputs, masks)

    return val_loss / len(dataloader), val_iou / len(dataloader)


if __name__ == "__main__":
    data_dir = "data/sample_train"
    image_size = 256
    batch_size = 2
    model_path = "models/best_mini_unet.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carga del dataset completo para validación
    image_list = sorted(os.listdir(os.path.join(data_dir, "images")))
    split_idx = int(len(image_list) * 0.8)
    val_images = image_list[split_idx:]

    dataset = RoadSegmentationDataset(
        images_dir=os.path.join(data_dir, "images"),
        masks_dir=os.path.join(data_dir, "masks"),
        transform=get_transforms(image_size=image_size)
    )

    val_loader = DataLoader(dataset, batch_size=batch_size)

    model = MiniUNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    criterion = nn.BCELoss()

    val_loss, val_iou = evaluate(model, val_loader, criterion, device)
    print(f"\nEvaluation Results:")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation IoU : {val_iou:.4f}")
