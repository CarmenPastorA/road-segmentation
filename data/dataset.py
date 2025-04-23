# data/dataset.py

import os
import cv2
import torch
from torch.utils.data import Dataset
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class RoadSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        self.image_ids = sorted([
            f for f in os.listdir(images_dir) if f.endswith("_sat.jpg")
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        mask_id = image_id.replace("_sat.jpg", "_mask.png")

        image_path = os.path.join(self.images_dir, image_id)
        mask_path = os.path.join(self.masks_dir, mask_id)

        # Load image and mask with OpenCV
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 128).astype('float32')  # Binarize mask

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)  # Add channel dimension

        return image, mask
