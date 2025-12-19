import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# =========================
# CBAM
# =========================
class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return torch.sigmoid(out).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return torch.sigmoid(x)


# =========================
# ConvNeXt + CBAM FER Model
# =========================
class ConvNeXtCBAMEmoteNet(nn.Module):
    """
    使用 ConvNeXt 作為 Backbone
    適用於表情辨識（FER）
    """

    def __init__(self, num_classes=7, backbone="convnext_tiny"):
        super().__init__()

        # ----------------------------------------------------
        # 1) 載入 ConvNeXt（timm 官方支援）
        # ----------------------------------------------------
        self.backbone = timm.create_model(
            backbone,
            pretrained=True,
            num_classes=0   # 移除分類頭
        )

        feat_dim = self.backbone.num_features  # convnext_tiny = 768

        # ----------------------------------------------------
        # 2) CBAM（接在最後一層 feature map）
        # ----------------------------------------------------
        self.cbam = CBAM(feat_dim)

        # ----------------------------------------------------
        # 3) 分類頭
        # ----------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

        # 初始化
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # ConvNeXt 特徵圖 [B, C, H, W]
        feat = self.backbone.forward_features(x)

        # CBAM 注意力
        feat = self.cbam(feat)

        # Global Average Pooling
        feat = feat.mean(dim=[2, 3])

        out = self.classifier(feat)
        return out
