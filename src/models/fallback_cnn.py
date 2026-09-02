"""
Model: FallbackCNN (MobileNetV3-Small)
========================================
Used when MediaPipe fails to detect landmarks (poor lighting, occlusion).
Takes a cropped, resized hand ROI image as input.

Input : (B, 3, 224, 224)  — normalized RGB hand crop
Output: (B, num_classes)

MobileNetV3-Small: 2.5M params, ~15ms CPU inference.
Pre-trained on ImageNet, fine-tuned on hand crops.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
import numpy as np


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────

class FallbackCNN(nn.Module):
    """
    MobileNetV3-Small with a custom classification head.
    Fine-tune last 2 stages + classifier for fast convergence.

    Args:
        num_classes: number of sign classes
        pretrained:  load ImageNet weights (recommended)
    """

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()

        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)

        # Freeze all but the last 2 InvertedResidual blocks + classifier
        all_layers = list(backbone.features.children())
        for layer in all_layers[:-2]:
            for param in layer.parameters():
                param.requires_grad = False

        self.features    = backbone.features
        self.avgpool     = backbone.avgpool

        # Custom head replacing the default MobileNetV3 classifier
        self.classifier = nn.Sequential(
            nn.Linear(576, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) normalized image tensor
        Returns:
            logits: (B, num_classes)
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=-1)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────
# Image preprocessing for inference
# ──────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

inference_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


def crop_hand_roi(frame: np.ndarray, results,
                   pad: float = 0.2) -> np.ndarray | None:
    """
    Extract a padded hand bounding-box crop from a MediaPipe result.
    Used to feed into FallbackCNN when landmark quality is low.

    Args:
        frame:   BGR frame from OpenCV
        results: mediapipe Hands result object
        pad:     fractional padding around the hand bounding box
    Returns:
        (H, W, 3) RGB crop or None if no hand detected
    """
    if not results.multi_hand_landmarks:
        return None

    h, w = frame.shape[:2]
    lm   = results.multi_hand_landmarks[0]

    xs = [p.x for p in lm.landmark]
    ys = [p.y for p in lm.landmark]

    x1 = int(max(0, (min(xs) - pad) * w))
    y1 = int(max(0, (min(ys) - pad) * h))
    x2 = int(min(w, (max(xs) + pad) * w))
    y2 = int(min(h, (max(ys) + pad) * h))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def prepare_image(crop_rgb: np.ndarray) -> torch.Tensor:
    """Convert a (H, W, 3) numpy RGB array to a (1, 3, 224, 224) tensor."""
    return inference_transform(crop_rgb).unsqueeze(0)


# ──────────────────────────────────────────────
# Sanity check
# ──────────────────────────────────────────────

if __name__ == "__main__":
    model = FallbackCNN(num_classes=36, pretrained=False)
    x     = torch.randn(2, 3, 224, 224)
    out   = model(x)
    print(f"Output shape : {out.shape}")            # (2, 36)
    print(f"Trainable params: {model.param_count:,}")
