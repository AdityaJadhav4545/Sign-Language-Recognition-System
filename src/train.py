"""
Module 4: Training Pipeline
============================
Single entry-point for training both static and dynamic models.
- Stratified train/val split
- WeightedRandomSampler for class imbalance
- CosineAnnealing LR schedule
- Label smoothing
- W&B logging
- Auto ONNX export on new best checkpoint

Usage:
    python src/train.py --config configs/static.yaml  --model static
    python src/train.py --config configs/dynamic.yaml --model dynamic
"""

import os
import sys
import time
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from dataset import StaticLandmarkDataset, DynamicLandmarkDataset, make_dataloaders
from models.static_mlp   import StaticGestureNet
from models.dynamic_lstm import DynamicGestureNet

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        print(f"[train] GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("[train] Running on CPU")
    return torch.device("cpu")


def build_model(model_type: str, num_classes: int,
                cfg: dict) -> nn.Module:
    if model_type == "static":
        return StaticGestureNet(
            num_classes=num_classes,
            dropout=cfg.get("dropout", 0.3)
        )
    elif model_type == "dynamic":
        return DynamicGestureNet(
            num_classes=num_classes,
            hidden=cfg.get("hidden_size", 128),
            layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.3)
        )
    raise ValueError(f"Unknown model type: {model_type}")


def export_onnx(model: nn.Module, model_type: str,
                export_dir: str, device: torch.device,
                seq_len: int = 30):
    """Export model to ONNX with dynamic batch axis."""
    os.makedirs(export_dir, exist_ok=True)
    model.eval()
    path = os.path.join(export_dir, f"{model_type}.onnx")

    if model_type == "static":
        dummy = torch.randn(1, 63).to(device)
        input_names  = ["landmarks"]
        dynamic_axes = {"landmarks": {0: "batch"}, "logits": {0: "batch"}}
    else:
        dummy = torch.randn(1, seq_len, 63).to(device)
        input_names  = ["sequence"]
        dynamic_axes = {"sequence": {0: "batch"}, "logits": {0: "batch"}}

    torch.onnx.export(
        model, dummy, path,
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True
    )
    print(f"[train] ONNX exported → {path}")


# ──────────────────────────────────────────────
# Training / Evaluation loops
# ──────────────────────────────────────────────

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    loss_fn: nn.Module,
                    device: torch.device) -> tuple[float, float]:
    """Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = correct = total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss   = loss_fn(logits, y)
        loss.backward()

        # Gradient clipping (helps LSTM stability)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += len(y)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             loss_fn: nn.Module,
             device: torch.device) -> tuple[float, float, list, list]:
    """Returns (avg_loss, accuracy, all_preds, all_labels)."""
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []

    for x, y in loader:
        x, y   = x.to(device), y.to(device)
        logits = model(x)
        loss   = loss_fn(logits, y)

        preds = logits.argmax(1)
        total_loss += loss.item() * len(y)
        correct    += (preds == y).sum().item()
        total      += len(y)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


# ──────────────────────────────────────────────
# Confusion matrix plot
# ──────────────────────────────────────────────

def plot_confusion_matrix(labels: list, preds: list,
                           classes: list, save_path: str):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[train] Confusion matrix saved → {save_path}")


# ──────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────

def train(cfg: dict, model_type: str):
    device      = get_device()
    classes     = cfg["classes"]
    num_classes = len(classes)

    # ── DataLoaders ──
    if model_type == "static":
        ds_cls  = StaticLandmarkDataset
        ds_kw   = {}
    else:
        ds_cls  = DynamicLandmarkDataset
        ds_kw   = {"seq_len": cfg.get("sequence_length", 30)}

    tr_loader, val_loader = make_dataloaders(
        root=cfg["data_root"],
        classes=classes,
        dataset_cls=ds_cls,
        batch_size=cfg["batch"],
        val_split=cfg["val_split"],
        num_workers=0,
        **ds_kw
    )

    # ── Model ──
    model = build_model(model_type, num_classes, cfg).to(device)
    print(f"[train] Model: {type(model).__name__}  "
          f"Params: {model.param_count:,}")

    # ── Optimizer / LR schedule / Loss ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 1e-4)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=1e-6
    )
    loss_fn = nn.CrossEntropyLoss(
        label_smoothing=cfg.get("label_smoothing", 0.1)
    )

    # ── W&B ──
    use_wandb = WANDB_AVAILABLE and os.environ.get("WANDB_API_KEY")
    if use_wandb:
        wandb.init(project="sign-language-recognition",
                   name=f"{model_type}_{time.strftime('%m%d_%H%M')}",
                   config=cfg)

    # ── Training loop ──
    ckpt_dir = cfg.get("checkpoint_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_acc = 0.0
    history      = {"tr_loss": [], "val_loss": [], "tr_acc": [], "val_acc": []}

    print(f"\n[train] Starting {cfg['epochs']} epochs …\n")

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        tr_loss,  tr_acc,  _,         _       = train_one_epoch(
            model, tr_loader, optimizer, loss_fn, device), None, None, None
        # Unpack properly
        tr_loss, tr_acc = tr_loss[0], tr_loss[1]

        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, val_loader, loss_fn, device)

        scheduler.step()
        dt = time.time() - t0

        # Book-keeping
        history["tr_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["tr_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:03d}/{cfg['epochs']}  "
              f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"lr={lr:.2e}  {dt:.1f}s")

        if use_wandb:
            wandb.log({"epoch": epoch, "tr_loss": tr_loss, "tr_acc": tr_acc,
                       "val_loss": val_loss, "val_acc": val_acc, "lr": lr})

        # Save best checkpoint + ONNX
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(ckpt_dir, f"best_{model_type}.pt")
            torch.save({
                "epoch":      epoch,
                "val_acc":    val_acc,
                "classes":    classes,
                "model_type": model_type,
                "state_dict": model.state_dict()
            }, ckpt_path)
            print(f"  ✓ New best val_acc={val_acc:.4f} → {ckpt_path}")
            export_onnx(model, model_type,
                        cfg.get("export_dir", "exports"), device,
                        cfg.get("sequence_length", 30))

    # ── Final report ──
    print(f"\n[train] Best val accuracy: {best_val_acc:.4f}")
    print("\n[train] Classification report (last epoch):\n")
    print(classification_report(val_labels, val_preds,
                                target_names=classes, digits=3))

    plot_confusion_matrix(
        val_labels, val_preds, classes,
        save_path=os.path.join(ckpt_dir, f"cm_{model_type}.png")
    )

    # ── Learning curve ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["tr_loss"], label="Train")
    ax1.plot(history["val_loss"], label="Val")
    ax1.set(title="Loss", xlabel="Epoch", ylabel="CrossEntropy")
    ax1.legend()
    ax2.plot(history["tr_acc"], label="Train")
    ax2.plot(history["val_acc"], label="Val")
    ax2.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ckpt_dir, f"curves_{model_type}.png"), dpi=150)
    plt.close()

    if use_wandb:
        wandb.finish()


# ──────────────────────────────────────────────
# Epoch helper fix (clean tuple unpacking)
# ──────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss   = loss_fn(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += len(y)
    return total_loss / total, correct / total, None, None


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLR Training")
    parser.add_argument("--config", default="configs/static.yaml")
    parser.add_argument("--model",  choices=["static", "dynamic"],
                        default="static")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, args.model)
