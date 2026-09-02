"""
Module 2: Preprocessing
========================
Landmark normalization and augmentation pipeline.
Works entirely in coordinate space — no image I/O.

Key transforms:
  1. Root-relative  → subtract wrist (idx 0) → position invariant
  2. Scale norm     → divide by max L2 distance → size invariant
  3. Augmentation   → jitter, rotation, flip in landmark space
"""

import numpy as np
from pathlib import Path
import os
import glob
from tqdm import tqdm


# ──────────────────────────────────────────────
# Normalization
# ──────────────────────────────────────────────

def normalize_landmarks(lm: np.ndarray) -> np.ndarray:
    """
    Root-relative + unit-scale normalization.

    Args:
        lm: (21, 3) float32 raw landmarks from MediaPipe
    Returns:
        (63,) float32 normalized feature vector
    """
    lm = lm.copy()
    lm -= lm[0]                                        # wrist to origin
    scale = np.max(np.linalg.norm(lm, axis=1)) + 1e-6 # largest distance
    lm /= scale                                        # unit scale → [-1, 1]
    return lm.flatten().astype(np.float32)             # (63,)


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Normalize a sequence of landmark frames.

    Args:
        seq: (T, 21, 3) float32
    Returns:
        (T, 63) float32
    """
    T = seq.shape[0]
    out = np.zeros((T, 63), dtype=np.float32)
    for i in range(T):
        out[i] = normalize_landmarks(seq[i])
    return out


# ──────────────────────────────────────────────
# Augmentation — landmark space only
# ──────────────────────────────────────────────

def rotate_z(lm: np.ndarray, deg: float) -> np.ndarray:
    """Rotate landmarks around the Z axis (camera-facing plane)."""
    theta = np.radians(deg)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ], dtype=np.float32)
    return (R @ lm.T).T


def augment_landmarks(lm: np.ndarray,
                       p_flip:   float = 0.5,
                       jitter:   float = 0.02,
                       rot_deg:  float = 15.0,
                       p_scale:  float = 0.5,
                       scale_r:  float = 0.1) -> np.ndarray:
    """
    Augment a single (21, 3) landmark frame.
    All operations are in normalized coordinate space.

    Args:
        lm:       (21, 3) landmark array (before normalization)
        p_flip:   probability of horizontal mirror flip
        jitter:   ±max uniform noise per coordinate
        rot_deg:  ±max rotation in degrees around Z axis
        p_scale:  probability of applying scale jitter
        scale_r:  ±range of scale factor jitter
    Returns:
        (21, 3) augmented landmarks
    """
    lm = lm.copy()

    # 1. Gaussian jitter — simulates sensor noise / slight tremor
    lm += np.random.uniform(-jitter, jitter, lm.shape).astype(np.float32)

    # 2. Z-axis rotation — simulates hand tilt in the image plane
    angle = np.random.uniform(-rot_deg, rot_deg)
    lm = rotate_z(lm, angle)

    # 3. Scale jitter — simulates different hand sizes / distances
    if np.random.rand() < p_scale:
        factor = 1.0 + np.random.uniform(-scale_r, scale_r)
        lm *= factor

    # 4. Mirror flip — left ↔ right hand
    #    NOTE: in ASL some signs are NOT mirror-symmetric (e.g. J, Z).
    #    Set p_flip=0 for those classes or train separate models.
    if np.random.rand() < p_flip:
        lm[:, 0] *= -1

    return lm


def augment_sequence(seq: np.ndarray, **kwargs) -> np.ndarray:
    """
    Augment a full (T, 21, 3) sequence with consistent transforms
    (same flip/rotation applied to every frame — temporal consistency).
    """
    T = seq.shape[0]
    seq = seq.copy()

    # Sample once and apply consistently across all frames
    p_flip  = kwargs.get("p_flip", 0.5)
    jitter  = kwargs.get("jitter", 0.015)
    rot_deg = kwargs.get("rot_deg", 10.0)

    do_flip  = np.random.rand() < p_flip
    angle    = np.random.uniform(-rot_deg, rot_deg)
    noise    = np.random.uniform(-jitter, jitter, seq.shape).astype(np.float32)

    for i in range(T):
        seq[i] += noise[i]
        seq[i]  = rotate_z(seq[i], angle)
        if do_flip:
            seq[i, :, 0] *= -1

    return seq


# ──────────────────────────────────────────────
# Offline batch preprocessing (optional cache)
# ──────────────────────────────────────────────

def preprocess_and_cache(src_root: str, dst_root: str,
                          n_augments: int = 3):
    """
    Read raw .npy landmarks → normalize → save augmented copies.
    Run once to pre-generate an augmented cache on disk.

    Args:
        src_root:   e.g. data/landmarks/{CLASS}/*.npy
        dst_root:   e.g. data/augmented/{CLASS}/*.npy
        n_augments: number of augmented copies per original sample
    """
    classes = sorted(os.listdir(src_root))
    print(f"[preprocess] Caching {len(classes)} classes with {n_augments}x augment …")

    for cls in tqdm(classes, desc="Classes"):
        src_cls = os.path.join(src_root, cls)
        dst_cls = os.path.join(dst_root, cls)
        os.makedirs(dst_cls, exist_ok=True)

        for path in glob.glob(f"{src_cls}/*.npy"):
            lm  = np.load(path)              # (21, 3)
            stem = Path(path).stem

            # Save original normalized
            feat = normalize_landmarks(lm)
            np.save(os.path.join(dst_cls, f"{stem}_orig.npy"), feat)

            # Save augmented copies
            for k in range(n_augments):
                aug  = augment_landmarks(lm)
                feat = normalize_landmarks(aug)
                np.save(os.path.join(dst_cls, f"{stem}_aug{k}.npy"), feat)

    print("[preprocess] Done.")


# ──────────────────────────────────────────────
# Statistics & quality check
# ──────────────────────────────────────────────

def dataset_stats(data_root: str):
    """
    Print per-class sample counts and flag imbalance.
    """
    classes = sorted(os.listdir(data_root))
    counts  = {}
    for cls in classes:
        n = len(glob.glob(f"{data_root}/{cls}/*.npy"))
        counts[cls] = n

    total   = sum(counts.values())
    avg     = total / len(counts)
    min_c   = min(counts.values())
    max_c   = max(counts.values())

    print(f"\n{'Class':<8} {'Count':>8}")
    print("-" * 18)
    for cls, n in sorted(counts.items(), key=lambda x: x[1]):
        flag = " ← LOW" if n < avg * 0.5 else ""
        print(f"{cls:<8} {n:>8}{flag}")
    print("-" * 18)
    print(f"{'Total':<8} {total:>8}")
    print(f"{'Min':<8} {min_c:>8}")
    print(f"{'Max':<8} {max_c:>8}")
    print(f"{'Avg':<8} {avg:>8.0f}")
    imbalance = max_c / (min_c + 1e-6)
    print(f"\nImbalance ratio: {imbalance:.1f}x  "
          f"{'(use WeightedRandomSampler)' if imbalance > 2 else '(OK)'}\n")
    return counts


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SLR Preprocessing")
    parser.add_argument("--mode", choices=["cache", "stats"], default="stats")
    parser.add_argument("--src",  default="data/landmarks")
    parser.add_argument("--dst",  default="data/augmented")
    parser.add_argument("--n_aug", type=int, default=3)
    args = parser.parse_args()

    if args.mode == "cache":
        preprocess_and_cache(args.src, args.dst, args.n_aug)
    elif args.mode == "stats":
        dataset_stats(args.src)
