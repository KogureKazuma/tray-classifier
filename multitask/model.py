"""
複数種類のトレーに対応するマルチタスク分類モデル

EfficientNet-B0 のバックボーンを共有し、2つのヘッドを持つ:
  - tray_type_head  : トレーの種類を分類
  - cleanliness_head: clean (残渣なし) / dirty (残渣あり) を分類
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

CLEANLINESS_NAMES = ["clean", "dirty"]


class MultiTaskTrayClassifier(nn.Module):
    def __init__(self, num_tray_types: int, freeze_base: bool = True):
        super().__init__()
        backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        in_features = backbone.classifier[1].in_features

        if freeze_base:
            for p in self.features.parameters():
                p.requires_grad = False

        self.dropout = nn.Dropout(p=0.3)
        self.tray_type_head = nn.Linear(in_features, num_tray_types)
        self.cleanliness_head = nn.Linear(in_features, len(CLEANLINESS_NAMES))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.tray_type_head(x), self.cleanliness_head(x)

    def unfreeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = True
