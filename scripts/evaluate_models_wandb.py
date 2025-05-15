"""
evaluate_cached_models_wandb.py

Evaluates multiple trained MiniUNet and MiniUNetPlus models using CachedRoadDataset and logs results to Weights & Biases.
"""

import torch
import wandb
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from models.unet_variants import get_unet_variant
from shared.metrics import compute_iou, compute_dice

# Dataset
class CachedRoadDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_path):
        self.images, self.masks = torch.load(tensor_path)
        if self.masks.ndim == 3:
            self.masks = self.masks.unsqueeze(1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].float() / 255.0
        mask = (self.masks[idx] > 0).float()
        if image.ndim == 2:
            image = image.unsqueeze(0)
        return image, mask


# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    ious, dices = [], []
    example_images = []

    with torch.no_grad():
        for idx, (images, masks) in enumerate(tqdm(dataloader, desc="Evaluating")):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = (outputs > 0.5).float()

            ious.append(compute_iou(preds, masks))
            dices.append(compute_dice(preds, masks))

            if idx == 0:
                for i in range(min(2, images.size(0))):
                    example_images.append({
                        "input": images[i].cpu(),
                        "prediction": preds[i].cpu(),
                        "ground_truth": masks[i].cpu()
                    })

    return np.mean(ious), np.mean(dices), example_images


# Main model evaluation configuration
model_configs = {
    "miniunet_cache": {
        "model_type": "MiniUNet",
        "path": "models/unet_checkpoints/best_miniunet_cache.pth",
        "val_split": "data/cache/val_split.pt",
        "epochs": 30,
        "image_size": "original",
        "augmentations": "none"
    },
    "miniunet_cache_HVFlip_256": {
        "model_type": "MiniUNet",
        "path": "models/unet_checkpoints/best_miniunet_cache_HVFlip_256.pth",
        "val_split": "data/cache/val_split_HVFlip_256.pt",
        "epochs": 30,
        "image_size": "256x256",
        "augmentations": "H+V Flip"
    },
    "miniunet_plus": {
        "model_type": "MiniUNetPlus",
        "path": "models/unet_checkpoints/best_miniunet_plus.pth",
        "val_split": "data/cache/val_split.pt",
        "epochs": 30,
        "image_size": "original",
        "augmentations": "none"
    },
    "miniunet_plus_HVFlip_256": {
        "model_type": "MiniUNetPlus",
        "path": "models/unet_checkpoints/best_miniunet_plus_HVFlip_256.pth",
        "val_split": "data/cache/val_split_HVFlip_256.pt",
        "epochs": 30,
        "image_size": "256x256",
        "augmentations": "H+V Flip"
    },
    "miniunet_plus_HVFlip_256_50e": {
        "model_type": "MiniUNetPlus",
        "path": "models/unet_checkpoints/best_miniunet_plus_HVFlip_256_50e.pth",
        "val_split": "data/cache/val_split_HVFlip_256.pt",
        "epochs": 50,
        "image_size": "256x256",
        "augmentations": "H+V Flip"
    },
    "attunet_HVFlip_256": {
        "model_type": "attunet", 
        "path": "models/unet_checkpoints/best_attunet_HVFlip_256.pth",
        "val_split": "data/cache/val_split_HVFlip_256.pt",
        "epochs": 50,
        "image_size": "256x256",
        "augmentations": "H+V Flip"
}
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "attunet_HVFlip_256"
    config = model_configs[model_name]

    print(f"Evaluating: {model_name}")

    wandb.init(
        project="road-segmentation",
        name=f"eval_{model_name}",
        config={
            "model": config["model_type"],
            "checkpoint": config["path"],
            "val_split": config["val_split"],
            "epochs": config["epochs"],
            "image_size": config["image_size"],
            "augmentations": config["augmentations"],
            "batch_size": 8,
            "loss": "BCE + Dice",
            "threshold": 0.5
        },
        reinit=True
    )

    model = get_unet_variant(config["model_type"]).to(device)
    model.load_state_dict(torch.load(config["path"], map_location=device))

    val_dataset = CachedRoadDataset(config["val_split"])
    val_loader = DataLoader(val_dataset, batch_size=8)

    iou, dice, examples = evaluate(model, val_loader, device)

    wandb.log({"IoU": iou, "Dice": dice})
    for idx, ex in enumerate(examples):
        wandb.log({
            f"example_{idx}/input": wandb.Image(ex["input"]),
            f"example_{idx}/prediction": wandb.Image(ex["prediction"]),
            f"example_{idx}/ground_truth": wandb.Image(ex["ground_truth"]),
        })

    wandb.finish()
    print(f"Finished: {model_name}")


if __name__ == "__main__":
    main()
