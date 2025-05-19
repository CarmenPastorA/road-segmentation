# scripts/train.py

"""
Training script for MiniUNet on the DeepGlobe Road Extraction Dataset.

- Binary Cross Entropy loss
- Optimizer: Adam (LR = 1e-3)
- Image size: 256x256
- Batch size: 2
- Metrics: IoU
- Trains for 10 epochs and saves the best model based on validation IoU

Logs are saved to logs/train_log.txt
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.mini_unet import MiniUNet
from data.dataset import RoadSegmentationDataset
from data.transforms import get_transforms
import logging

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    handlers=[
        logging.FileHandler(f"{log_dir}/train_log.txt"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()


# Calculate the IoU score for the predicted and target masks.
def iou_score(preds, targets, threshold=0.5, eps=1e-6):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = ((preds + targets) >= 1).float().sum(dim=(1, 2, 3))
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()

# Train the model for one epoch.
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0
    epoch_iou = 0.0

    for images, masks in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_iou += iou_score(outputs.detach(), masks)

    return epoch_loss / len(dataloader), epoch_iou / len(dataloader)

# Validate the model on the validation set.
def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_iou = 0.0

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            val_loss += loss.item()
            val_iou += iou_score(outputs, masks)

    return val_loss / len(dataloader), val_iou / len(dataloader)


def main():
    # Config
    data_dir = "data/sample_train"
    batch_size = 2
    image_size = 256
    num_epochs = 10
    learning_rate = 1e-3
    best_model_path = "models/best_mini_unet.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("Starting training...")
    log.info(f"Using device: {device}")

    # Split train/val
    image_list = sorted(os.listdir(os.path.join(data_dir, "images")))
    num_samples = len(image_list)
    split_idx = int(num_samples * 0.8)
    train_images = image_list[:split_idx]
    val_images = image_list[split_idx:]

    # Helper function for dataset split
    def get_subset(image_list_subset):
        return RoadSegmentationDataset(
            images_dir=os.path.join(data_dir, "images"),
            masks_dir=os.path.join(data_dir, "masks"),
            transform=get_transforms(image_size=image_size),
        )

    train_dataset = get_subset(train_images)
    val_dataset = get_subset(val_images)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model
    model = MiniUNet(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_iou = 0.0

    for epoch in range(num_epochs):
        log.info(f"Epoch {epoch + 1}/{num_epochs}")

        train_loss, train_iou = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_iou = validate(model, val_loader, criterion, device)

        log.info(f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f}")
        log.info(f"Val   Loss: {val_loss:.4f} | Val   IoU: {val_iou:.4f}")

        # Save best model based on IoU
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), best_model_path)
            log.info(f"Best model saved! (IoU: {best_iou:.4f})")

    log.info("Training complete!")
    log.info(f"Best model saved at: {best_model_path}")


if __name__ == "__main__":
    main()
