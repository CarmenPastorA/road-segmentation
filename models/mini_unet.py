# models/unet.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
MiniUNet (U-Net variant)

This is a minimal U-Net-style architecture designed for binary segmentation tasks
like road detection from satellite images.

- Encoder: 3 double conv blocks (32 → 64 → 128)
- Bottleneck: 256 filters
- Decoder: 3 transposed conv + double conv blocks with skip connections
- Output: sigmoid-activated binary mask (1 channel)

Input: RGB image (3 channels), usually 256x256
Output: Mask with pixel values in [0, 1]
"""

class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class MiniUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # Down
        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)

        # Bottleneck
        self.bottleneck = DoubleConv(128, 256)

        # Up
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)

        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(64, 32)

        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(F.max_pool2d(x1, 2))
        x3 = self.enc3(F.max_pool2d(x2, 2))

        # Bottleneck
        x = self.bottleneck(F.max_pool2d(x3, 2))

        # Decoder
        x = self.up1(x)
        x = self.dec1(torch.cat([x, x3], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))

        x = self.up3(x)
        x = self.dec3(torch.cat([x, x1], dim=1))

        return torch.sigmoid(self.final_conv(x))  # output in [0, 1]
