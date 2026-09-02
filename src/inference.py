"""
Module 5: Real-Time Inference
==============================
Dual-mode inference pipeline:
  1. MediaPipe Hands → landmark extraction
  2. ONNX Runtime   → fast CPU inference (static or dynamic model)
  3. FallbackCNN    → activated when MediaPipe confidence is low
  4. PhraseBuilder  → debounce signs → phrase

Controls (in the webcam window):
  ESC     — quit
  c       — clear phrase
  b       — backspace last sign
  s       — screenshot
  SPACE   — pause/resume

Usage:
    python src/inference.py --config configs/static.yaml
    python src/inference.py --config configs/dynamic.yaml --model dynamic
"""

import sys
import os
import time
import argparse
import yaml
import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.dirname(__file__))
from preprocess    import normalize_landmarks, normalize_sequence
from phrase_builder import PhraseBuilder, WordPhraseBuilder
from models.fallback_cnn import FallbackCNN, crop_hand_roi, prepare_image


# ──────────────────────────────────────────────
# ONNX Inference Engines
# ──────────────────────────────────────────────

class StaticInferenceEngine:
    """ONNX-based static gesture classifier."""

    def __init__(self, model_path: str, classes: list,
                 conf_thresh: float = 0.75):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.classes    = classes
        self.thresh     = conf_thresh

    def predict(self, lm: np.ndarray) -> tuple[str | None, float]:
        """
        Args:
            lm: (21, 3) normalized landmark array
        Returns:
            (class_name or None, confidence)
        """
        feat = normalize_landmarks(lm)[None]   # (1, 63)
        logits = self.session.run(None, {self.input_name: feat})[0][0]
        probs  = _softmax(logits)
        top    = probs.argmax()
        conf   = float(probs[top])

        if conf >= self.thresh:
            return self.classes[top], conf
        return None, conf


class DynamicInferenceEngine:
    """ONNX-based dynamic gesture (word/phrase) classifier."""

    def __init__(self, model_path: str, classes: list,
                 seq_len: int = 30, conf_thresh: float = 0.70):
        self.session    = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.classes    = classes
        self.seq_len    = seq_len
        self.thresh     = conf_thresh
        self._buffer    = []   # accumulates (21,3) frames

    def push_frame(self, lm: np.ndarray | None) -> tuple[str | None, float]:
        """
        Feed one frame. Returns a prediction only once seq_len frames
        have accumulated; otherwise returns (None, 0.0).
        """
        if lm is None:
            self._buffer.append(np.zeros((21, 3), dtype=np.float32))
        else:
            self._buffer.append(lm)

        if len(self._buffer) < self.seq_len:
            return None, 0.0

        # Predict on the current window
        seq  = np.stack(self._buffer[-self.seq_len:], axis=0)  # (T, 21, 3)
        feat = normalize_sequence(seq)[None].astype(np.float32) # (1, T, 63)
        logits = self.session.run(None, {self.input_name: feat})[0][0]
        probs  = _softmax(logits)
        top    = probs.argmax()
        conf   = float(probs[top])

        # Slide window (don't wait for full new sequence every time)
        self._buffer = self._buffer[self.seq_len // 2:]

        if conf >= self.thresh:
            return self.classes[top], conf
        return None, conf

    def reset(self):
        self._buffer.clear()


# ──────────────────────────────────────────────
# Fallback CNN Engine
# ──────────────────────────────────────────────

class FallbackEngine:
    """PyTorch MobileNetV3 fallback for poor-landmark frames."""

    def __init__(self, ckpt_path: str, classes: list,
                 conf_thresh: float = 0.65):
        self.classes = classes
        self.thresh  = conf_thresh
        self.model   = FallbackCNN(num_classes=len(classes), pretrained=False)

        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            self.model.load_state_dict(ckpt.get("state_dict", ckpt))
        self.model.eval()

    def predict(self, frame: np.ndarray, results) -> tuple[str | None, float]:
        crop = crop_hand_roi(frame, results, pad=0.25)
        if crop is None:
            return None, 0.0
        tensor = prepare_image(crop)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=-1)[0].numpy()
        top  = probs.argmax()
        conf = float(probs[top])
        if conf >= self.thresh:
            return self.classes[top], conf
        return None, conf


# ──────────────────────────────────────────────
# HUD drawing helpers
# ──────────────────────────────────────────────

mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

FONT         = cv2.FONT_HERSHEY_DUPLEX
FONT_SM      = cv2.FONT_HERSHEY_SIMPLEX
GREEN        = (0, 220, 100)
WHITE        = (230, 230, 230)
GRAY         = (120, 120, 120)
DARK         = (20, 20, 20)
ACCENT       = (0, 180, 255)


def draw_hud(frame: np.ndarray, sign: str | None, conf: float,
             phrase: str, paused: bool, fps: float,
             stability: float, source: str = "landmark") -> np.ndarray:
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 70), DARK, -1)

    if sign:
        cv2.putText(frame, sign, (20, 52), FONT, 1.8, GREEN, 2, cv2.LINE_AA)
        # Confidence bar
        bar_x, bar_y, bar_w, bar_h = 100, 20, 160, 12
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (60, 60, 60), -1)
        filled = int(bar_w * conf)
        color  = (0, 200, 80) if conf > 0.85 else (0, 150, 255)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + filled, bar_y + bar_h), color, -1)
        cv2.putText(frame, f"{conf:.0%}", (bar_x + bar_w + 8, bar_y + 11),
                    FONT_SM, 0.55, WHITE, 1, cv2.LINE_AA)

    # Stability ring indicator
    stab_filled = int(stability * 360)
    cv2.ellipse(frame, (w - 45, 35), (22, 22), -90,
                0, stab_filled, ACCENT, 3, cv2.LINE_AA)

    # FPS
    cv2.putText(frame, f"{fps:.0f}fps", (w - 90, 65),
                FONT_SM, 0.55, GRAY, 1, cv2.LINE_AA)

    # Source indicator
    src_color = GREEN if source == "landmark" else (0, 150, 255)
    cv2.putText(frame, source, (w - 180, 65),
                FONT_SM, 0.5, src_color, 1, cv2.LINE_AA)

    # Phrase bar
    phrase_y = h - 15
    cv2.rectangle(frame, (0, h - 55), (w, h), DARK, -1)
    cv2.putText(frame, phrase[-60:] if len(phrase) > 60 else phrase,
                (15, phrase_y), FONT_SM, 0.85, WHITE, 1, cv2.LINE_AA)

    # Paused overlay
    if paused:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, "PAUSED", (w // 2 - 90, h // 2),
                    FONT, 2, WHITE, 3, cv2.LINE_AA)

    # Controls hint (small, bottom-right)
    hint = "ESC:quit  c:clear  b:back  SPACE:pause  s:screenshot"
    cv2.putText(frame, hint, (10, h - 62),
                FONT_SM, 0.38, GRAY, 1, cv2.LINE_AA)

    return frame


# ──────────────────────────────────────────────
# Main webcam loop
# ──────────────────────────────────────────────

def run_webcam(cfg: dict, model_type: str = "static"):
    classes     = cfg["classes"]
    conf_thresh = cfg.get("conf_threshold", 0.75)
    hold_frames = cfg.get("hold_frames", 12)
    export_dir  = cfg.get("export_dir", "exports")
    ckpt_dir    = cfg.get("checkpoint_dir", "checkpoints")

    # ── Engines ──
    static_path  = os.path.join(export_dir, "static.onnx")
    dynamic_path = os.path.join(export_dir, "dynamic.onnx")
    fallback_path = os.path.join(ckpt_dir, "best_fallback.pt")

    if model_type == "static":
        engine = StaticInferenceEngine(static_path, classes, conf_thresh)
        phrase_buf = PhraseBuilder(
            hold_frames=hold_frames,
            space_frames=cfg.get("space_frames", 20)
        )
    else:
        engine = DynamicInferenceEngine(
            dynamic_path, classes,
            seq_len=cfg.get("sequence_length", 30),
            conf_thresh=conf_thresh
        )
        phrase_buf = WordPhraseBuilder()

    # Fallback (optional — loads silently if checkpoint exists)
    fallback = None
    if os.path.exists(fallback_path):
        fallback = FallbackEngine(fallback_path, classes)
        print("[inference] Fallback CNN loaded")

    # ── MediaPipe ──
    mp_hands_sol = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=cfg.get("min_detection_confidence", 0.8),
        min_tracking_confidence=0.7
    )

    # ── Webcam ──
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)

    print("[inference] Press ESC to quit, c to clear, b to backspace, SPACE to pause")

    paused      = False
    fps_ema     = 30.0
    prev_t      = time.perf_counter()
    screenshot_n = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)   # mirror view

        # FPS smoothing
        now    = time.perf_counter()
        dt     = max(now - prev_t, 1e-6)
        fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt)
        prev_t = now

        if paused:
            draw_hud(frame, None, 0, phrase_buf.get_phrase(),
                     True, fps_ema, 0)
            cv2.imshow("Sign Language Recognition", frame)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                break
            if key == ord(' '):
                paused = False
            continue

        # ── MediaPipe processing ──
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands_sol.process(rgb)

        # Draw skeleton
        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_lm, mp.solutions.hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

        # ── Predict ──
        lm = None
        if results.multi_hand_landmarks:
            lm = np.array(
                [[p.x, p.y, p.z]
                 for p in results.multi_hand_landmarks[0].landmark],
                dtype=np.float32
            )

        sign, conf, source = None, 0.0, "landmark"

        if model_type == "static":
            if lm is not None:
                sign, conf = engine.predict(lm)
            elif fallback is not None:
                sign, conf = fallback.predict(frame, results)
                source = "fallback"
            phrase_buf.push(sign)

        else:  # dynamic
            sign, conf = engine.push_frame(lm)
            if sign:
                phrase_buf.push(sign)
            elif sign is None and lm is None:
                phrase_buf.push(None)

        # ── HUD ──
        draw_hud(
            frame,
            sign, conf,
            phrase_buf.get_phrase(),
            paused, fps_ema,
            phrase_buf.stability_pct() if model_type == "static" else 0.0,
            source
        )

        cv2.imshow("Sign Language Recognition", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:   # ESC
            break
        elif key == ord('c'):
            phrase_buf.clear()
            print("[inference] Phrase cleared")
        elif key == ord('b'):
            phrase_buf.backspace()
        elif key == ord(' '):
            paused = True
        elif key == ord('s'):
            path = f"screenshot_{screenshot_n:03d}.png"
            cv2.imwrite(path, frame)
            print(f"[inference] Screenshot saved → {path}")
            screenshot_n += 1

    cap.release()
    cv2.destroyAllWindows()
    mp_hands_sol.close()


# ──────────────────────────────────────────────
# Math helper
# ──────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLR Real-Time Inference")
    parser.add_argument("--config", default="configs/static.yaml")
    parser.add_argument("--model",  choices=["static", "dynamic"],
                        default="static")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_webcam(cfg, args.model)
