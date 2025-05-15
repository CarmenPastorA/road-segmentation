import torch
import torch.nn as nn
import torch.nn.functional as F

# DoubleConv con BatchNorm y Dropout opcional
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


# Mini U-Net básico (filtros 32-64-128-256)
class MiniUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.bottleneck = DoubleConv(128, 256)

        self.up1 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = DoubleConv(128, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec3 = DoubleConv(64, 32)
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(F.max_pool2d(x1, 2))
        x3 = self.enc3(F.max_pool2d(x2, 2))
        x = self.bottleneck(F.max_pool2d(x3, 2))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, x3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))
        x = self.up3(x)
        x = self.dec3(torch.cat([x, x1], dim=1))
        return torch.sigmoid(self.final_conv(x))


# MiniUNetPlus: más filtros + dropout
class MiniUNetPlus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, dropout=0.2):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.bottleneck = DoubleConv(256, 512)

        self.up1 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec1 = DoubleConv(512, 256, dropout=dropout)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = DoubleConv(256, 128, dropout=dropout)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec3 = DoubleConv(128, 64, dropout=dropout)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(F.max_pool2d(x1, 2))
        x3 = self.enc3(F.max_pool2d(x2, 2))
        x = self.bottleneck(F.max_pool2d(x3, 2))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, x3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))
        x = self.up3(x)
        x = self.dec3(torch.cat([x, x1], dim=1))
        return torch.sigmoid(self.final_conv(x))
    
# Convolutional block used in AttU_Net
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# Attention gate for skip connections
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# Attention U-Net architecture
class AttU_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super().__init__()
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]  # 64, 128, 256, 512, 1024

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder path
        self.Conv1 = ConvBlock(img_ch, filters[0])
        self.Conv2 = ConvBlock(filters[0], filters[1])
        self.Conv3 = ConvBlock(filters[1], filters[2])
        self.Conv4 = ConvBlock(filters[2], filters[3])
        self.Conv5 = ConvBlock(filters[3], filters[4])  # Bottleneck

        # Decoder path with attention
        self.Up5 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
        self.Att5 = Attention_block(filters[3], filters[3], filters[2])
        self.Up_conv5 = ConvBlock(filters[4], filters[3])

        self.Up4 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.Att4 = Attention_block(filters[2], filters[2], filters[1])
        self.Up_conv4 = ConvBlock(filters[3], filters[2])

        self.Up3 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.Att3 = Attention_block(filters[1], filters[1], filters[0])
        self.Up_conv3 = ConvBlock(filters[2], filters[1])

        self.Up2 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.Att2 = Attention_block(filters[0], filters[0], 32)
        self.Up_conv2 = ConvBlock(filters[1], filters[0])

        self.Conv_1x1 = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # ---- Encoder ----
        x1 = self.Conv1(x)
        x2 = self.Conv2(self.Maxpool1(x1))
        x3 = self.Conv3(self.Maxpool2(x2))
        x4 = self.Conv4(self.Maxpool3(x3))
        x5 = self.Conv5(self.Maxpool4(x4))

        # ---- Decoder ----
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = self.Up_conv5(torch.cat((x4, d5), dim=1))

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = self.Up_conv4(torch.cat((x3, d4), dim=1))

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = self.Up_conv3(torch.cat((x2, d3), dim=1))

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = self.Up_conv2(torch.cat((x1, d2), dim=1))

        out = self.Conv_1x1(d2)
        return torch.sigmoid(out)


# Factory para facilitar instanciación por nombre
def get_unet_variant(name: str, **kwargs):
    name = name.lower()
    if name == "miniunet":
        return MiniUNet(**kwargs)
    elif name == "miniunetplus":
        return MiniUNetPlus(**kwargs)
    elif name == "attunet":
        return AttU_Net(img_ch=kwargs.get("in_channels", 3), output_ch=kwargs.get("out_channels", 1))
    else:
        raise ValueError(f"Unknown model name: {name}")
