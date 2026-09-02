"""
Data Downloader  (v3 — webcam-first + protobuf fix)
====================================================
The Sign Language MNIST CSV files are stored via Git LFS on GitHub
and cannot be fetched via raw URLs without authentication.

This script instead:
  1. Fixes the protobuf conflict so MediaPipe works correctly
  2. Generates high-quality synthetic landmarks for ALL 36 classes
  3. Provides a guided webcam collector for real data (best quality)

Run order (recommended):
    python src/download_data.py --fix_protobuf   # fix mediapipe crash first
    python src/download_data.py --source synthetic  # 200 samples/class instantly
    python src/collect.py --mode static ...         # add real webcam data
    python src/download_data.py --stats
"""

import os, sys, subprocess, time, argparse
from pathlib import Path
import numpy as np

try:
    from tqdm import tqdm
    def progress(it, **kw): return tqdm(it, **kw)
except ImportError:
    def progress(it, desc="", total=None, **kw):
        print(f"  {desc}..."); return it


# ─────────────────────────────────────────────────────────────────────
# Fix 1: protobuf version conflict (causes MediaPipe AttributeError)
# ─────────────────────────────────────────────────────────────────────

def fix_protobuf():
    """
    The error:
        AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
        AttributeError: 'FieldDescriptor' object has no attribute 'label'

    Cause: mediapipe 0.10.x needs protobuf 3.20.x but pip installs 4.x by default.
    Fix:   downgrade protobuf to the version mediapipe requires.
    """
    print("[fix] Checking protobuf version ...")
    try:
        import google.protobuf
        version = google.protobuf.__version__
        major = int(version.split(".")[0])
        print(f"[fix] Current protobuf: {version}")
        if major >= 4:
            print("[fix] protobuf >= 4.x detected — downgrading to 3.20.3 ...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "protobuf==3.20.3", "--quiet"
            ])
            print("[fix] Done. Please restart your terminal and rerun your command.")
        else:
            print("[fix] protobuf version OK — no fix needed.")
    except Exception as e:
        print(f"[fix] Error: {e}")
        print("[fix] Try manually: pip install protobuf==3.20.3")


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_landmark(lm, out_dir, stem):
    ensure_dir(out_dir)
    np.save(os.path.join(out_dir, f"{stem}.npy"), lm.astype(np.float32))

def dataset_counts(root):
    counts = {}
    if not os.path.exists(root):
        return counts
    for cls in sorted(os.listdir(root)):
        p = os.path.join(root, cls)
        if not os.path.isdir(p): continue
        n = len(list(Path(p).glob("*.npy")))
        if n: counts[cls] = n
    return counts

def print_stats(root):
    counts = dataset_counts(root)
    if not counts:
        print(f"[stats] No data found in {root}"); return
    total = sum(counts.values())
    print(f"\n  {'Class':<8} {'Count':>8}\n  {'─'*18}")
    for cls, n in sorted(counts.items()):
        bar = "█" * min(20, n // 10)
        print(f"  {cls:<8} {n:>6}  {bar}")
    print(f"  {'─'*18}\n  {'TOTAL':<8} {total:>8}  ({len(counts)} classes)\n")

def verify_landmarks(root, n=10):
    import glob, random
    files = glob.glob(f"{root}/**/*.npy", recursive=True)
    if not files:
        print(f"[verify] No .npy files in {root}"); return
    sample = random.sample(files, min(n, len(files)))
    ok = fail = 0
    for p in sample:
        lm = np.load(p)
        if lm.shape == (21, 3) and np.isfinite(lm).all(): ok += 1
        else: print(f"  FAIL {p}"); fail += 1
    print(f"[verify] {ok+fail} files → {ok} OK  {fail} failed")
    if ok:
        lm = np.load(sample[0])
        print(f"         shape={lm.shape}  range=[{lm.min():.3f}, {lm.max():.3f}]")


# ─────────────────────────────────────────────────────────────────────
# Synthetic landmark generation  (pure numpy, no MediaPipe)
# ─────────────────────────────────────────────────────────────────────

def _hand_template():
    """Neutral open hand in MediaPipe skeleton proportions."""
    return np.array([
        [0.50,0.90,0.00],                            # 0 wrist
        [0.40,0.80,0.00],[0.30,0.72,0.00],           # 1-2  thumb CMC MCP
        [0.22,0.65,0.00],[0.15,0.58,0.00],           # 3-4  thumb IP TIP
        [0.42,0.70,0.00],[0.40,0.55,0.00],           # 5-6  index MCP PIP
        [0.40,0.42,0.00],[0.40,0.30,0.00],           # 7-8  index DIP TIP
        [0.50,0.68,0.00],[0.50,0.53,0.00],           # 9-10 middle
        [0.50,0.40,0.00],[0.50,0.27,0.00],           # 11-12
        [0.58,0.70,0.00],[0.60,0.55,0.00],           # 13-14 ring
        [0.61,0.43,0.00],[0.62,0.31,0.00],           # 15-16
        [0.66,0.73,0.00],[0.69,0.60,0.00],           # 17-18 pinky
        [0.70,0.50,0.00],[0.71,0.41,0.00],           # 19-20
    ], dtype=np.float32)

def _curl(lm, finger, amount):
    bases = [1,5,9,13,17]
    lm = lm.copy()
    b  = bases[finger]
    for j in range(1,4):
        lm[b+j,1] += amount * j * 0.08
        lm[b+j,0] += amount * j * 0.02
    return lm

# ASL hand configurations: list of (finger_index, curl_amount)
# finger 0=thumb 1=index 2=middle 3=ring 4=pinky
# curl  0.0=straight  1.0=fully curled
ASL_CONFIGS = {
    # ── Digits ──────────────────────────────────────────
    "0": [(0,0.5),(1,0.5),(2,0.5),(3,0.5),(4,0.5)],
    "1": [(0,0.7),(2,0.9),(3,0.9),(4,0.9)],
    "2": [(0,0.7),(3,0.9),(4,0.9)],
    "3": [(0,0.5),(3,0.9),(4,0.9)],
    "4": [(0,0.9)],
    "5": [],
    "6": [(1,0.9),(2,0.9),(3,0.9)],
    "7": [(1,0.9),(2,0.9),(4,0.9)],
    "8": [(1,0.9),(3,0.9),(4,0.9)],
    "9": [(2,0.9),(3,0.9),(4,0.9)],
    # ── Letters ─────────────────────────────────────────
    "A": [(0,0.3),(1,0.9),(2,0.9),(3,0.9),(4,0.9)],
    "B": [(0,0.9)],
    "C": [(0,0.3),(1,0.3),(2,0.3),(3,0.3),(4,0.3)],
    "D": [(0,0.4),(2,0.9),(3,0.9),(4,0.9)],
    "E": [(0,0.7),(1,0.7),(2,0.7),(3,0.7),(4,0.7)],
    "F": [(0,0.2),(1,0.6)],
    "G": [(0,0.2),(2,0.9),(3,0.9),(4,0.9)],
    "H": [(0,0.9),(3,0.9),(4,0.9)],
    "I": [(0,0.9),(1,0.9),(2,0.9),(3,0.9)],
    "J": [(0,0.9),(1,0.9),(2,0.9),(3,0.9)],
    "K": [(0,0.4),(3,0.9),(4,0.9)],
    "L": [(0,0.0),(1,0.0),(2,0.9),(3,0.9),(4,0.9)],
    "M": [(0,0.5),(1,0.8),(2,0.8),(3,0.8),(4,0.9)],
    "N": [(0,0.5),(1,0.8),(2,0.8),(3,0.9),(4,0.9)],
    "O": [(0,0.4),(1,0.4),(2,0.4),(3,0.4),(4,0.4)],
    "P": [(0,0.2),(1,0.0),(2,0.9),(3,0.9),(4,0.9)],
    "Q": [(0,0.1),(2,0.9),(3,0.9),(4,0.9)],
    "R": [(0,0.9),(3,0.9),(4,0.9)],
    "S": [(0,0.6),(1,0.9),(2,0.9),(3,0.9),(4,0.9)],
    "T": [(0,0.4),(1,0.8),(2,0.9),(3,0.9),(4,0.9)],
    "U": [(0,0.9),(3,0.9),(4,0.9)],
    "V": [(0,0.7),(3,0.9),(4,0.9)],
    "W": [(0,0.7),(4,0.9)],
    "X": [(0,0.9),(1,0.4),(2,0.9),(3,0.9),(4,0.9)],
    "Y": [(0,0.0),(1,0.9),(2,0.9),(3,0.9)],
    "Z": [(1,0.5),(2,0.9),(3,0.9),(4,0.9)],
}

def generate_synthetic(out_root="data/landmarks", n_per_class=300, classes=None):
    """
    Generate realistic synthetic MediaPipe-style landmarks.
    Uses 5x more variation than v1: scale, rotation, jitter, wrist offset,
    and per-finger length variation to simulate different hand shapes.
    """
    if classes is None:
        classes = list(ASL_CONFIGS.keys())

    print(f"\n[synthetic] Generating {n_per_class} samples × "
          f"{len(classes)} classes ...")

    rng   = np.random.default_rng(42)
    saved = {}

    for cls in classes:
        if cls not in ASL_CONFIGS:
            continue
        out_dir = os.path.join(out_root, cls)
        ensure_dir(out_dir)

        for i in range(n_per_class):
            lm = _hand_template().copy()

            # Apply ASL finger configuration
            for finger, amount in ASL_CONFIGS[cls]:
                # Randomise curl amount slightly (±15%) for variation
                actual = np.clip(amount + rng.normal(0, 0.08), 0, 1)
                lm = _curl(lm, finger, float(actual))

            # Per-finger length variation (simulate different hand sizes)
            for f in range(5):
                scale_f = rng.uniform(0.88, 1.12)
                base    = [1,5,9,13,17][f]
                pivot   = lm[base].copy()
                for j in range(1,4):
                    lm[base+j] = pivot + (lm[base+j] - pivot) * scale_f

            # Global noise
            lm += rng.normal(0, 0.010, lm.shape).astype(np.float32)

            # Random scale (±18%)
            c   = lm[0].copy()
            lm  = (lm - c) * rng.uniform(0.82, 1.18) + c

            # Random Z-axis rotation (±25°)
            th  = np.radians(rng.uniform(-25, 25))
            R   = np.array([[np.cos(th),-np.sin(th),0],
                             [np.sin(th), np.cos(th),0],
                             [0,0,1]], dtype=np.float32)
            lm  = ((R @ (lm - c).T).T + c).astype(np.float32)

            # Random wrist offset (simulate hand position in frame)
            lm += rng.uniform(-0.05, 0.05, (1, 3)).astype(np.float32)

            # Random mirror flip (left/right hand)
            if rng.random() < 0.4:
                lm[:,0] = 1.0 - lm[:,0]

            save_landmark(lm, out_dir, f"synth_{i:05d}")

        saved[cls] = n_per_class
        print(f"  {cls}: {n_per_class}", end="  ", flush=True)
        if (list(ASL_CONFIGS.keys()).index(cls) + 1) % 9 == 0:
            print()

    print(f"\n[synthetic] Done — {sum(saved.values()):,} landmarks saved.")


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SLR data helper — fix protobuf, generate synthetic data"
    )
    parser.add_argument("--fix_protobuf", action="store_true",
                        help="Fix the protobuf/mediapipe version conflict")
    parser.add_argument("--source", choices=["synthetic","digits","all"],
                        help="Generate synthetic landmarks")
    parser.add_argument("--stats",  action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--out",           default="data/landmarks")
    parser.add_argument("--n_per_class",   type=int, default=300,
                        help="Samples per class (default 300)")
    args = parser.parse_args()

    if args.fix_protobuf:
        fix_protobuf()
        sys.exit(0)

    if args.stats:
        print_stats(args.out); sys.exit(0)

    if args.verify:
        verify_landmarks(args.out); sys.exit(0)

    t0 = time.time()

    if args.source in ("synthetic", "all", "digits"):
        classes = list(ASL_CONFIGS.keys())  # all 36 classes
        generate_synthetic(args.out, args.n_per_class, classes)

    print(f"\nDone in {time.time()-t0:.1f}s")
    print_stats(args.out)
    verify_landmarks(args.out)
