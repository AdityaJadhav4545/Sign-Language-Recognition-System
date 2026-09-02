"""
Module 1: Data Collection
=========================
Collects hand landmark data from a webcam for each sign class.
Stores landmarks as .npy arrays (not images) — 100x smaller, faster to load.

Usage:
    python src/collect.py --mode static  --out data/landmarks
    python src/collect.py --mode dynamic --out data/sequences --seq_len 30
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import uuid
import argparse
import time
from pathlib import Path

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
STATIC_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [str(i) for i in range(10)]
SAMPLES_PER_CLASS = 300          # static: frames per class
SEQUENCE_LENGTH   = 30           # dynamic: frames per clip
SEQUENCES_PER_CLASS = 100        # dynamic: clips per word
MIN_DETECTION_CONF  = 0.85
COUNTDOWN_SECONDS   = 3

mp_hands  = mp.solutions.hands
mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


# ──────────────────────────────────────────────
# Landmark extraction helpers
# ──────────────────────────────────────────────

def extract_landmarks(results) -> np.ndarray | None:
    """
    Extract the first detected hand's landmarks.
    Returns ndarray of shape (21, 3) [x, y, z in [0,1]] or None.
    """
    if not results.multi_hand_landmarks:
        return None
    lm = results.multi_hand_landmarks[0]
    return np.array(
        [[p.x, p.y, p.z] for p in lm.landmark],
        dtype=np.float32
    )


def draw_overlay(frame: np.ndarray, results, label: str,
                 collected: int, total: int) -> np.ndarray:
    """Draw landmarks + HUD onto frame."""
    if results.multi_hand_landmarks:
        for hand_lm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )
    # HUD bar
    cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"Class: {label}   [{collected}/{total}]",
                (15, 40), cv2.FONT_HERSHEY_DUPLEX, 1.1,
                (0, 220, 100), 2, cv2.LINE_AA)
    return frame


def countdown(cap: cv2.VideoCapture, label: str, seconds: int = COUNTDOWN_SECONDS):
    """Show a countdown before recording starts."""
    for i in range(seconds, 0, -1):
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.rectangle(frame, (0, 0), (640, 480), (0, 0, 0), -1)
        cv2.putText(frame, f"Get ready: {label}", (80, 200),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, (255, 255, 255), 2)
        cv2.putText(frame, str(i), (290, 340),
                    cv2.FONT_HERSHEY_DUPLEX, 4, (0, 220, 100), 4)
        cv2.imshow("SLR — Data Collection", frame)
        cv2.waitKey(1000)


# ──────────────────────────────────────────────
# Static collection (single frames)
# ──────────────────────────────────────────────

def collect_static_class(label: str, out_dir: str,
                          n_samples: int = SAMPLES_PER_CLASS):
    """
    Collect N static landmark snapshots for one class.
    Skips frames where MediaPipe confidence is too low.
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    countdown(cap, label)
    collected = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=MIN_DETECTION_CONF,
        min_tracking_confidence=0.7
    ) as hands:

        while collected < n_samples:
            ok, frame = cap.read()
            if not ok:
                continue

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            lm      = extract_landmarks(results)

            if lm is not None:
                np.save(os.path.join(out_dir, f"{uuid.uuid4()}.npy"), lm)
                collected += 1

            frame = draw_overlay(frame, results, label, collected, n_samples)
            cv2.imshow("SLR — Data Collection", frame)
            if cv2.waitKey(1) == 27:  # ESC to skip class
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[collect] {label}: saved {collected} samples → {out_dir}")


def collect_all_static(out_root: str, classes: list,
                        n_samples: int = SAMPLES_PER_CLASS):
    """
    Iterate over all classes, press SPACE to start each,
    ESC to skip, 'q' to quit entirely.
    """
    cap = cv2.VideoCapture(0)

    print("\nControls: SPACE = start class | ESC = skip | q = quit\n")

    for cls in classes:
        out_dir = os.path.join(out_root, cls)
        existing = len(list(Path(out_dir).glob("*.npy"))) if os.path.exists(out_dir) else 0

        if existing >= n_samples:
            print(f"[collect] {cls}: already has {existing} samples, skipping.")
            continue

        # Wait for SPACE key
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            cv2.rectangle(frame, (0, 0), (640, 480), (20, 20, 20), -1)
            cv2.putText(frame, f"Next class: {cls}  ({existing} existing)",
                        (60, 200), cv2.FONT_HERSHEY_DUPLEX, 1.2, (200, 200, 200), 2)
            cv2.putText(frame, "SPACE to start  |  ESC to skip  |  q to quit",
                        (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 100, 100), 1)
            cv2.imshow("SLR — Data Collection", frame)
            key = cv2.waitKey(30) & 0xFF
            if key == ord(' '):
                break
            if key == 27:  # ESC
                out_dir = None
                break
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        if out_dir:
            cap.release()
            cv2.destroyAllWindows()
            collect_static_class(cls, out_dir, n_samples - existing)
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cap.release()
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────
# Dynamic collection (sequences)
# ──────────────────────────────────────────────

def collect_dynamic_class(label: str, out_dir: str,
                           n_sequences: int = SEQUENCES_PER_CLASS,
                           seq_len: int = SEQUENCE_LENGTH):
    """
    Collect N sequences of seq_len frames per dynamic class.
    Each file is shape (seq_len, 21, 3).
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    collected = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.7
    ) as hands:

        while collected < n_sequences:
            countdown(cap, label, seconds=2)
            sequence = []

            while len(sequence) < seq_len:
                ok, frame = cap.read()
                if not ok:
                    continue
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                lm      = extract_landmarks(results)

                if lm is not None:
                    sequence.append(lm)
                else:
                    sequence.append(np.zeros((21, 3), dtype=np.float32))

                frame = draw_overlay(frame, results, label,
                                     len(sequence), seq_len)
                cv2.putText(frame, f"Seq {collected+1}/{n_sequences}",
                            (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 200, 0), 1)
                cv2.imshow("SLR — Data Collection", frame)
                if cv2.waitKey(1) == 27:
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            arr = np.array(sequence, dtype=np.float32)  # (T, 21, 3)
            np.save(os.path.join(out_dir, f"{uuid.uuid4()}.npy"), arr)
            collected += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"[collect] {label}: saved {collected} sequences → {out_dir}")


# ──────────────────────────────────────────────
# Offline extraction from image dataset
# ──────────────────────────────────────────────

def extract_from_images(img_root: str, out_root: str,
                         min_conf: float = 0.7):
    """
    Offline: extract landmarks from the Kaggle ASL Alphabet dataset.
    Expects structure: img_root/{CLASS}/{image}.jpg

    Usage:
        python src/collect.py --mode offline_extract
                              --img_root data/raw
                              --out data/landmarks
    """
    import glob
    from tqdm import tqdm

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=min_conf
    ) as hands:

        all_paths = glob.glob(f"{img_root}/**/*.jpg", recursive=True)
        print(f"[extract] Found {len(all_paths)} images")
        skipped = 0

        for path in tqdm(all_paths, desc="Extracting landmarks"):
            label   = os.path.basename(os.path.dirname(path))
            img     = cv2.imread(path)
            if img is None:
                skipped += 1
                continue

            rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if not results.multi_hand_landmarks:
                skipped += 1
                continue

            lm = np.array(
                [[p.x, p.y, p.z] for p in results.multi_hand_landmarks[0].landmark],
                dtype=np.float32
            )
            out_dir = os.path.join(out_root, label)
            os.makedirs(out_dir, exist_ok=True)
            stem = Path(path).stem
            np.save(os.path.join(out_dir, f"{stem}.npy"), lm)

    print(f"[extract] Done. Skipped {skipped} images (no hand detected).")


# ──────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import yaml

    parser = argparse.ArgumentParser(description="SLR Data Collection")
    parser.add_argument("--mode", choices=["static", "dynamic", "offline_extract"],
                        default="static")
    parser.add_argument("--out",  default="data/landmarks")
    parser.add_argument("--img_root", default="data/raw",
                        help="Image root for offline_extract mode")
    parser.add_argument("--config", default="configs/static.yaml")
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_CLASS)
    parser.add_argument("--seq_len", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--n_seq",   type=int, default=SEQUENCES_PER_CLASS)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    classes = cfg["classes"]

    if args.mode == "static":
        collect_all_static(args.out, classes, args.samples)

    elif args.mode == "dynamic":
        for cls in classes:
            collect_dynamic_class(
                cls,
                os.path.join(args.out, cls),
                n_sequences=args.n_seq,
                seq_len=args.seq_len
            )

    elif args.mode == "offline_extract":
        extract_from_images(args.img_root, args.out)
