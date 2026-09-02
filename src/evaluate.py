"""
Evaluation Script
=================
Load a trained checkpoint and run a full evaluation:
  - Per-class accuracy, precision, recall, F1
  - Confusion matrix heatmap
  - Inference latency benchmark
  - Word Error Rate (for phrase-level evaluation)

Usage:
    python src/evaluate.py --config configs/static.yaml --model static
    python src/evaluate.py --config configs/dynamic.yaml --model dynamic
"""

import sys
import os
import time
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(__file__))
from dataset import (StaticLandmarkDataset, DynamicLandmarkDataset,
                     make_dataloaders)
from models.static_mlp   import StaticGestureNet
from models.dynamic_lstm import DynamicGestureNet
from preprocess import normalize_landmarks, normalize_sequence


# ──────────────────────────────────────────────
# PyTorch evaluation
# ──────────────────────────────────────────────

@torch.no_grad()
def evaluate_pytorch(model: nn.Module, loader: DataLoader,
                     device: torch.device) -> dict:
    model.eval()
    all_preds, all_labels = [], []

    for x, y in loader:
        x, y   = x.to(device), y.to(device)
        preds  = model(x).argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro")
    return {"acc": acc, "macro_f1": f1,
            "preds": all_preds, "labels": all_labels}


# ──────────────────────────────────────────────
# ONNX latency benchmark
# ──────────────────────────────────────────────

def benchmark_onnx(model_path: str, input_shape: tuple,
                   n_runs: int = 1000) -> dict:
    """
    Measure ONNX model latency on CPU.

    Args:
        model_path:  path to .onnx file
        input_shape: shape of single input (e.g. (1, 63) or (1, 30, 63))
        n_runs:      number of inference calls

    Returns:
        dict with mean_ms, p50_ms, p95_ms, p99_ms
    """
    session    = ort.InferenceSession(
        model_path, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    dummy      = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(20):
        session.run(None, {input_name: dummy})

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times)),
        "p50_ms":  float(np.percentile(times, 50)),
        "p95_ms":  float(np.percentile(times, 95)),
        "p99_ms":  float(np.percentile(times, 99)),
    }


# ──────────────────────────────────────────────
# Confusion matrix plot
# ──────────────────────────────────────────────

def plot_confusion_matrix(labels: list, preds: list,
                           classes: list, save_path: str,
                           normalize: bool = False):
    cm = confusion_matrix(labels, preds)
    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-6)
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        fmt = "d"
        title = "Confusion Matrix (counts)"

    n = len(classes)
    fig_size = max(10, n * 0.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size - 1))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=classes, yticklabels=classes,
                linewidths=0.4, ax=ax, annot_kws={"size": 8})
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",      fontsize=11)
    ax.set_title(title,        fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[eval] Confusion matrix → {save_path}")


# ──────────────────────────────────────────────
# Per-class accuracy bar chart
# ──────────────────────────────────────────────

def plot_per_class_accuracy(labels: list, preds: list,
                             classes: list, save_path: str):
    cm = confusion_matrix(labels, preds)
    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-6)

    fig, ax = plt.subplots(figsize=(max(10, len(classes) * 0.45), 5))
    colors = ["#1D9E75" if a >= 0.9 else "#EF9F27" if a >= 0.75
              else "#D85A30" for a in per_class_acc]
    ax.bar(classes, per_class_acc, color=colors)
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, label="90%")
    ax.set(xlabel="Class", ylabel="Accuracy",
           title="Per-class Accuracy", ylim=(0, 1.05))
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[eval] Per-class accuracy → {save_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def run_evaluation(cfg: dict, model_type: str):
    classes  = cfg["classes"]
    ckpt_dir = cfg.get("checkpoint_dir", "checkpoints")
    exp_dir  = cfg.get("export_dir", "exports")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ──
    ckpt_path = os.path.join(ckpt_dir, f"best_{model_type}.pt")
    if not os.path.exists(ckpt_path):
        print(f"[eval] Checkpoint not found: {ckpt_path}")
        print("  → Train the model first with: python src/train.py")
        return

    ckpt = torch.load(ckpt_path, map_location=device)
    if model_type == "static":
        model = StaticGestureNet(num_classes=len(classes),
                                 dropout=cfg.get("dropout", 0.3))
        ds_cls = StaticLandmarkDataset
        ds_kw  = {}
        onnx_shape = (1, 63)
    else:
        model = DynamicGestureNet(
            num_classes=len(classes),
            hidden=cfg.get("hidden_size", 128),
            layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.3)
        )
        ds_cls = DynamicLandmarkDataset
        ds_kw  = {"seq_len": cfg.get("sequence_length", 30)}
        onnx_shape = (1, cfg.get("sequence_length", 30), 63)

    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    print(f"[eval] Loaded epoch {ckpt.get('epoch','?')}  "
          f"(saved val_acc={ckpt.get('val_acc', 0):.4f})")

    # ── DataLoader (validation split only) ──
    _, val_loader = make_dataloaders(
        root=cfg["data_root"],
        classes=classes,
        dataset_cls=ds_cls,
        batch_size=cfg["batch"],
        val_split=cfg["val_split"],
        **ds_kw
    )

    # ── PyTorch evaluation ──
    results = evaluate_pytorch(model, val_loader, device)
    print(f"\n[eval] Accuracy : {results['acc']:.4f}")
    print(f"[eval] Macro F1 : {results['macro_f1']:.4f}\n")
    print(classification_report(results["labels"], results["preds"],
                                target_names=classes, digits=3))

    # ── Plots ──
    plot_confusion_matrix(
        results["labels"], results["preds"], classes,
        save_path=os.path.join(ckpt_dir, f"cm_{model_type}.png")
    )
    plot_confusion_matrix(
        results["labels"], results["preds"], classes,
        save_path=os.path.join(ckpt_dir, f"cm_{model_type}_norm.png"),
        normalize=True
    )
    plot_per_class_accuracy(
        results["labels"], results["preds"], classes,
        save_path=os.path.join(ckpt_dir, f"per_class_{model_type}.png")
    )

    # ── ONNX latency benchmark ──
    onnx_path = os.path.join(exp_dir, f"{model_type}.onnx")
    if os.path.exists(onnx_path):
        lat = benchmark_onnx(onnx_path, onnx_shape, n_runs=500)
        print(f"\n[eval] ONNX latency ({model_type}) — 500 runs on CPU:")
        print(f"  mean  : {lat['mean_ms']:.2f} ms")
        print(f"  p50   : {lat['p50_ms']:.2f} ms")
        print(f"  p95   : {lat['p95_ms']:.2f} ms")
        print(f"  p99   : {lat['p99_ms']:.2f} ms")
        real_time_ok = lat["p95_ms"] < (1000 / 25)  # 25fps budget
        print(f"  Real-time (<40ms): {'✓ YES' if real_time_ok else '✗ NO'}")
    else:
        print(f"[eval] ONNX model not found at {onnx_path} — skipping latency test")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLR Evaluation")
    parser.add_argument("--config", default="configs/static.yaml")
    parser.add_argument("--model",  choices=["static", "dynamic"],
                        default="static")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_evaluation(cfg, args.model)
