"""
Dataset Module
==============
PyTorch Dataset implementations for static and dynamic gestures.
Handles on-the-fly augmentation and stratified splitting.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from preprocess import normalize_landmarks, augment_landmarks
from preprocess import normalize_sequence, augment_sequence


# ──────────────────────────────────────────────
# Static Dataset — single (21,3) frames
# ──────────────────────────────────────────────

class StaticLandmarkDataset(Dataset):
    """
    Loads pre-saved (21, 3) .npy landmark files.
    Applies normalization and optional augmentation on __getitem__.

    Directory structure expected:
        root/
          A/  *.npy
          B/  *.npy
          ...
    """

    def __init__(self, root: str, classes: list,
                 augment: bool = False,
                 indices: list = None):
        self.augment = augment
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples = []   # list of (filepath, label_idx)

        for cls in classes:
            cls_dir = os.path.join(root, cls)
            if not os.path.exists(cls_dir):
                continue
            for p in glob.glob(f"{cls_dir}/*.npy"):
                self.samples.append((p, self.class_to_idx[cls]))

        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        lm = np.load(path)           # (21, 3)

        if self.augment:
            lm = augment_landmarks(lm)

        feat = normalize_landmarks(lm)   # (63,)
        return torch.from_numpy(feat), torch.tensor(label, dtype=torch.long)

    @property
    def labels(self) -> list:
        return [s[1] for s in self.samples]


# ──────────────────────────────────────────────
# Dynamic Dataset — sequences (T, 21, 3)
# ──────────────────────────────────────────────

class DynamicLandmarkDataset(Dataset):
    """
    Loads pre-saved (T, 21, 3) .npy sequence files.
    Pads/truncates to a fixed sequence length.

    Directory structure expected:
        root/
          hello/  *.npy
          thanks/ *.npy
          ...
    """

    def __init__(self, root: str, classes: list,
                 seq_len: int = 30,
                 augment: bool = False,
                 indices: list = None):
        self.seq_len = seq_len
        self.augment = augment
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.samples = []

        for cls in classes:
            cls_dir = os.path.join(root, cls)
            if not os.path.exists(cls_dir):
                continue
            for p in glob.glob(f"{cls_dir}/*.npy"):
                self.samples.append((p, self.class_to_idx[cls]))

        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

    def __len__(self) -> int:
        return len(self.samples)

    def _pad_or_truncate(self, seq: np.ndarray) -> np.ndarray:
        """Ensure sequence has exactly self.seq_len frames."""
        T = seq.shape[0]
        if T >= self.seq_len:
            return seq[:self.seq_len]
        # Pad with zeros
        pad = np.zeros((self.seq_len - T, 21, 3), dtype=np.float32)
        return np.concatenate([seq, pad], axis=0)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        seq = np.load(path)          # (T, 21, 3)

        if self.augment:
            seq = augment_sequence(seq)

        seq  = self._pad_or_truncate(seq)   # (seq_len, 21, 3)
        feat = normalize_sequence(seq)       # (seq_len, 63)

        return (
            torch.from_numpy(feat),
            torch.tensor(label, dtype=torch.long)
        )

    @property
    def labels(self) -> list:
        return [s[1] for s in self.samples]


# ──────────────────────────────────────────────
# Weighted sampler (handles class imbalance)
# ──────────────────────────────────────────────

def build_weighted_sampler(dataset) -> WeightedRandomSampler:
    """
    Creates a WeightedRandomSampler that up-samples rare classes.
    Ensures each batch has approximately uniform class distribution.
    """
    labels      = torch.tensor(dataset.labels)
    class_count = torch.bincount(labels)
    # Inverse frequency weighting
    weights     = 1.0 / (class_count.float() + 1e-6)
    sample_w    = weights[labels]
    return WeightedRandomSampler(
        weights=sample_w,
        num_samples=len(sample_w),
        replacement=True
    )


# ──────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────

def make_dataloaders(root: str, classes: list,
                     dataset_cls,
                     batch_size: int = 128,
                     val_split:  float = 0.15,
                     num_workers: int = 0,
                     **dataset_kwargs):
    """
    Creates stratified train/val DataLoaders.

    Args:
        root:         path to landmark root directory
        classes:      list of class names
        dataset_cls:  StaticLandmarkDataset or DynamicLandmarkDataset
        batch_size:   batch size
        val_split:    fraction for validation
        num_workers:  DataLoader workers (0 = main process)
        **dataset_kwargs: extra args forwarded to dataset_cls

    Returns:
        (train_loader, val_loader)
    """
    full_ds = dataset_cls(root, classes, augment=False, **dataset_kwargs)
    n       = len(full_ds)
    labels  = full_ds.labels

    tr_idx, val_idx = train_test_split(
        range(n),
        test_size=val_split,
        stratify=labels,
        random_state=42
    )

    tr_ds  = dataset_cls(root, classes, augment=True,
                         indices=list(tr_idx), **dataset_kwargs)
    val_ds = dataset_cls(root, classes, augment=False,
                         indices=list(val_idx), **dataset_kwargs)

    sampler = build_weighted_sampler(tr_ds)

    tr_loader = DataLoader(
        tr_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    print(f"[dataset] Train: {len(tr_ds):,}  Val: {len(val_ds):,}  "
          f"Classes: {len(classes)}  Batch: {batch_size}")
    return tr_loader, val_loader


# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import yaml

    with open("configs/static.yaml") as f:
        cfg = yaml.safe_load(f)

    tr, val = make_dataloaders(
        root=cfg["data_root"],
        classes=cfg["classes"],
        dataset_cls=StaticLandmarkDataset,
        batch_size=cfg["batch"],
        val_split=cfg["val_split"]
    )

    x, y = next(iter(tr))
    print(f"Batch x: {x.shape}  y: {y.shape}  dtype: {x.dtype}")
