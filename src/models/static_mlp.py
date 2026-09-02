"""
Model: StaticGestureNet (MLP)
==============================
Input : (B, 63)  — normalized landmark vector
Output: (B, num_classes)  — logits

Architecture:
  Linear(63→256) → BN → ReLU → Dropout
  Linear(256→128) → BN → ReLU → Dropout
  Linear(128→64)  → ReLU
  Linear(64→num_classes)

Trains in <2 min on CPU. Inference <3ms per frame.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StaticGestureNet(nn.Module):
    """
    Lightweight MLP for static ASL sign classification.

    Args:
        num_classes: number of output classes (36 for A-Z + 0-9)
        dropout:     dropout rate (default 0.3)
    """

    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()

        self.net = nn.Sequential(
            # Block 1
            nn.Linear(63, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            # Block 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),

            # Block 3 — no BN/dropout before final
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),

            # Classifier head
            nn.Linear(64, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming init for ReLU activations."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 63) normalized landmark features
        Returns:
            logits: (B, num_classes)
        """
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities. Useful for thresholding at inference."""
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=-1)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────
# Sanity check
# ──────────────────────────────────────────────

if __name__ == "__main__":
    model = StaticGestureNet(num_classes=36)
    x     = torch.randn(8, 63)
    out   = model(x)
    print(f"Output shape : {out.shape}")          # (8, 36)
    print(f"Parameters   : {model.param_count:,}")  # ~54K
