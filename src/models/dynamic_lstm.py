"""
Model: DynamicGestureNet (Bi-LSTM)
====================================
Input : (B, T, 63)  — sequence of normalized landmark vectors
Output: (B, num_classes)  — logits

Architecture:
  Linear(63→128) — frame embedding
  BiLSTM(128, hidden=128, layers=2)
  Last-timestep pooling → (B, 256)
  Linear(256→64) → ReLU → Linear(64→num_classes)

Use T=30 (1 second at 30fps) by default.
Inference: ~8ms per sequence on CPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicGestureNet(nn.Module):
    """
    Bidirectional LSTM for dynamic ASL sign/word classification.

    Args:
        num_classes: number of output word/phrase classes
        hidden:      LSTM hidden size per direction (total = 2×hidden)
        layers:      number of LSTM layers
        dropout:     dropout rate applied between LSTM layers
        embed_dim:   dimension of per-frame embedding
    """

    def __init__(self, num_classes: int,
                 hidden:    int = 128,
                 layers:    int = 2,
                 dropout:   float = 0.3,
                 embed_dim: int = 128):
        super().__init__()

        # Per-frame feature embedding
        self.frame_embed = nn.Sequential(
            nn.Linear(63, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True)
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=True
        )

        lstm_out_dim = hidden * 2  # bidirectional

        # Attention pooling over time dimension
        self.attn = nn.Linear(lstm_out_dim, 1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 63) normalized landmark sequences
        Returns:
            logits: (B, num_classes)
        """
        B, T, _ = x.shape

        # Embed each frame independently
        x = self.frame_embed(x)              # (B, T, embed_dim)

        # LSTM over time
        out, _ = self.lstm(x)                # (B, T, hidden*2)

        # Soft attention pooling — weight each timestep
        attn_w = self.attn(out)              # (B, T, 1)
        attn_w = torch.softmax(attn_w, dim=1)
        ctx    = (out * attn_w).sum(dim=1)   # (B, hidden*2)

        return self.classifier(ctx)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=-1)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────
# Sanity check
# ──────────────────────────────────────────────

if __name__ == "__main__":
    model = DynamicGestureNet(num_classes=30, hidden=128, layers=2)
    x     = torch.randn(4, 30, 63)   # batch=4, T=30, features=63
    out   = model(x)
    print(f"Output shape : {out.shape}")           # (4, 30)
    print(f"Parameters   : {model.param_count:,}") # ~380K
