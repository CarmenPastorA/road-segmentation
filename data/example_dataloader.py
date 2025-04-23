# example_dataloader.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch.utils.data import DataLoader
from data.dataset import RoadSegmentationDataset
from data.transforms import get_transforms

if __name__ == "__main__":
    dataset = RoadSegmentationDataset(
        images_dir="data/sample_train/images",
        masks_dir="data/sample_train/masks",
        transform=get_transforms(image_size=256)
    )

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    print(f"Dataset loaded with {len(dataset)} samples.")
    for i, (image, mask) in enumerate(dataloader):
        print(f"Batch {i+1} — Image shape: {image.shape}, Mask shape: {mask.shape}")
        if i == 1:
            break  # Just preview 2 batches
