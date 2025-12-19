import torch
import torch.nn as nn
from torchvision import models


class ViT_FER(nn.Module):
    def __init__(self, num_classes=7, freeze_backbone=False):
        super().__init__()

        # 1️⃣ ViT backbone
        self.vit = models.vit_b_16(
            weights=models.ViT_B_16_Weights.IMAGENET1K_V1
        )

        # 2️⃣ 改分類頭（FER 專用）
        in_dim = self.vit.heads.head.in_features

        self.vit.heads = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # 3️⃣（可選）freeze ViT backbone
        if freeze_backbone:
            for p in self.vit.parameters():
                p.requires_grad = False
            for p in self.vit.heads.parameters():
                p.requires_grad = True

    def forward(self, x):
        """
        x: [B, 3, 224, 224]
        """
        return self.vit(x)
