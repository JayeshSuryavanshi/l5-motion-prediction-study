"""Learned trajectory prediction models."""

import timm
import torch
from torch import nn


class HistoryMLP(nn.Module):
    """E2: multi-mode trajectory regression from agent history alone, no map."""

    def __init__(
        self,
        feature_dim: int,
        num_modes: int = 3,
        num_future: int = 50,
        hidden_dim: int = 512,
        depth: int = 3,
    ):
        super().__init__()
        self.num_modes = num_modes
        self.num_future = num_future
        layers: list[nn.Module] = [nn.LayerNorm(feature_dim)]
        in_dim = feature_dim
        for _ in range(depth):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        self.body = nn.Sequential(*layers)
        self.trajectory_head = nn.Linear(hidden_dim, num_modes * num_future * 2)
        self.confidence_head = nn.Linear(hidden_dim, num_modes)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """features (B, feature_dim) -> predictions (B, K, T, 2), log-confidences (B, K)."""
        hidden = self.body(features)
        predictions = self.trajectory_head(hidden).view(
            -1, self.num_modes, self.num_future, 2
        )
        log_confidences = torch.log_softmax(self.confidence_head(hidden), dim=1)
        return predictions, log_confidences


class RasterCNN(nn.Module):
    """E3/E4: multi-mode trajectory regression from a rasterized BEV image."""

    def __init__(
        self,
        in_channels: int,
        backbone: str = "resnet18",
        num_modes: int = 3,
        num_future: int = 50,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_modes = num_modes
        self.num_future = num_future
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, in_chans=in_channels, num_classes=0
        )
        feature_dim = self.backbone.num_features
        self.trajectory_head = nn.Linear(feature_dim, num_modes * num_future * 2)
        self.confidence_head = nn.Linear(feature_dim, num_modes)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """images (B, C, H, W) in [0, 1] -> predictions (B, K, T, 2), log-confidences (B, K)."""
        hidden = self.backbone(images)
        predictions = self.trajectory_head(hidden).view(
            -1, self.num_modes, self.num_future, 2
        )
        log_confidences = torch.log_softmax(self.confidence_head(hidden), dim=1)
        return predictions, log_confidences
