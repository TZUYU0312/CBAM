import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm  # Vision Mamba 使用 timm 提供的模型

class CBAM(nn.Module):
    """
    CBAM (Convolutional Block Attention Module) 注意力機制模組
    結合通道注意力和空間注意力，用於強化特徵圖的重要區域
    
    參數:
        channels (int): 輸入特徵圖的通道數
        reduction_ratio (int): 通道注意力中的降維比例，預設為16
    """
    def __init__(self, channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x) * x  # 給每個通道一個權重
        x = self.spatial_attention(x) * x  # 給每個像素一個權重
        return x #回傳特徵圖

class ChannelAttention(nn.Module):
    """
    通道注意力模組
    使用平均池化和最大池化來獲取通道間的關係
    
    參數:
        channels (int): 輸入特徵圖的通道數
        reduction_ratio (int): 降維比例，用於減少計算量
    """
    def __init__(self, channels, reduction_ratio):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全域平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全域最大池化
        
        # 多層感知器，用於特徵轉換
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels)
        )

    def forward(self, x):
        """
        前向傳播
        
        參數:
            x (Tensor): 輸入特徵圖
            
        返回:
            Tensor: 通道注意力權重
        """
        b, c, _, _ = x.size()
        # 計算平均池化和最大池化的特徵
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out  # 特徵融合
        return torch.sigmoid(out).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    """
    空間注意力模組
    使用通道維度的平均值和最大值來獲取空間注意力圖
    """
    def __init__(self):
        super(SpatialAttention, self).__init__()
        # 7x7 卷積核用於空間特徵提取
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        """
        前向傳播
        
        參數:
            x (Tensor): 輸入特徵圖
            
        返回:
            Tensor: 空間注意力權重
        """
        # 計算通道維度的平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)  # 特徵串接
        x = self.conv(x)  # 空間特徵提取
        return torch.sigmoid(x)  # 轉換為注意力權重

class ResidualBlock(nn.Module): #定義一個「殘差區塊」，是 ResNet 結構的核心組件，加上 CBAM 注意力機制
    """
    殘差區塊
    包含兩個卷積層和一個CBAM注意力模組，並具有捷徑連接
    """
    def __init__(self, in_channels, out_channels, stride=1): #初始化：輸入/輸出通道數、卷積步長，並建立卷積層、BN、注意力模組。
        super(ResidualBlock, self).__init__()
        # 第一個卷積層 (3x3 卷積 (帶 stride，可改變特徵圖大小)，再接 BatchNorm → 作用：提取特徵。
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 第二個卷積層 (第二個 3x3 卷積，輸入和輸出通道相同，保持大小 → 作用：進一步提取更深層次的特徵。
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                              kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # CBAM注意力模組
        self.cbam = CBAM(out_channels)
        
        # 捷徑連接 (保證殘差結構中「輸入 (identity)」和「卷積後輸出 (out)」維度相同)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 
                         kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """
        前向傳播
        
        參數:
            x (Tensor): 輸入特徵圖
            
        返回:
            Tensor: 經過殘差區塊處理後的特徵圖
        """
        identity = x #保存輸入，作為捷徑
        
        # 主要路徑
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)  # 注意力機制
        
        # 捷徑連接
        out += self.shortcut(identity)
        out = F.relu(out)
        
        return out




class EnhancedVisionMambaEmoteNet(nn.Module):
    """
    使用 Vision Mamba (Vim) 作為 Backbone
    適用於表情辨識（FER），取代 Transformer (ViT)
    """

    def __init__(self, num_classes=7, model_name="vim_small_patch16_224"):
        super(EnhancedVisionMambaEmoteNet, self).__init__()

        # ----------------------------------------------------
        # 1) 載入 Vision Mamba (Vim-S / Vim-T / Vim-B)
        # ----------------------------------------------------
        self.vim = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0      # 不要用預設分類頭
        )

        # Vision Mamba 輸出維度 (tiny=384, small=768, base=1024)
        vim_dim = self.vim.num_features

        # ----------------------------------------------------
        # 2) 自訂分類頭（比 ViT 時更輕量，效果更好）
        # ----------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(vim_dim, 512),
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
        feat = self.vim(x)  # Vision Mamba 的 feature
        out = self.classifier(feat)
        return out