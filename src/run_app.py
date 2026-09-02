"""
run_app.py — Single command to launch the full SignSense web app
================================================================
Usage:
    python src/run_app.py
    python src/run_app.py --config configs/static.yaml
    python src/run_app.py --port 8080

What it does:
    1. Checks trained model exists (checkpoints/best_static.pt)
    2. Exports ONNX if not already done
    3. Copies static.onnx → website/static.onnx (served as static file)
    4. Starts a Flask server that serves website/ and the /predict API
    5. Opens http://localhost:5000/camera in your browser automatically
"""

import os
import sys
import shutil
import argparse
import threading
import webbrowser
import time
import yaml
import json
import base64
import numpy as np

# ── optional imports with friendly errors ──────────────────────────
try:
    from flask import Flask, send_from_directory, jsonify, request, Response
    from flask_cors import CORS
except ImportError:
    print("[error] Flask not installed. Run:")
    print("        pip install flask flask-cors")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("[error] PyTorch not installed. Run:")
    print("        pip install torch --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(1)

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

# ──────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEBSITE_DIR = os.path.join(ROOT, "website")
EXPORTS_DIR = os.path.join(ROOT, "exports")
CKPT_DIR    = os.path.join(ROOT, "checkpoints")
ONNX_SRC    = os.path.join(EXPORTS_DIR, "static.onnx")
ONNX_WEB    = os.path.join(WEBSITE_DIR, "static.onnx")


# ──────────────────────────────────────────────────────────────────
# Step 1 — Export ONNX if needed
# ──────────────────────────────────────────────────────────────────

def ensure_onnx(cfg: dict):
    """Export ONNX from checkpoint if exports/static.onnx doesn't exist."""
    if os.path.exists(ONNX_SRC):
        print(f"[run_app] ONNX model found: {ONNX_SRC}")
        return True

    ckpt_path = os.path.join(CKPT_DIR, "best_static.pt")
    if not os.path.exists(ckpt_path):
        print(f"[run_app] ERROR: No trained model found at {ckpt_path}")
        print("          Train first:  python src/train.py --config configs/static.yaml")
        return False

    print(f"[run_app] Exporting ONNX from {ckpt_path} ...")
    try:
        from models.static_mlp import StaticGestureNet
        ckpt    = torch.load(ckpt_path, map_location="cpu")
        classes = ckpt.get("classes", cfg["classes"])
        model   = StaticGestureNet(num_classes=len(classes))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        os.makedirs(EXPORTS_DIR, exist_ok=True)
        dummy = torch.randn(1, 63)
        torch.onnx.export(
            model, dummy, ONNX_SRC,
            input_names=["landmarks"],
            output_names=["logits"],
            dynamic_axes={"landmarks": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17
        )
        print(f"[run_app] ONNX exported → {ONNX_SRC}")
        return True
    except Exception as e:
        print(f"[run_app] ONNX export failed: {e}")
        return False


# ──────────────────────────────────────────────────────────────────
# Step 2 — Copy ONNX to website/
# ──────────────────────────────────────────────────────────────────

def copy_model_to_website():
    """Copy exports/static.onnx → website/static.onnx."""
    if not os.path.exists(ONNX_SRC):
        return False
    os.makedirs(WEBSITE_DIR, exist_ok=True)
    shutil.copy2(ONNX_SRC, ONNX_WEB)
    size_kb = os.path.getsize(ONNX_WEB) // 1024
    print(f"[run_app] Model copied → {ONNX_WEB}  ({size_kb} KB)")
    return True


# ──────────────────────────────────────────────────────────────────
# Step 3 — Flask server
# ──────────────────────────────────────────────────────────────────

def create_app(cfg: dict) -> Flask:
    app = Flask(__name__, static_folder=WEBSITE_DIR)
    CORS(app)  # allow cross-origin (needed for ONNX fetch from browser)

    classes = cfg.get("classes", [])

    # ── Serve website files ────────────────────────────────────────

    @app.route("/")
    def index():
        return send_from_directory(WEBSITE_DIR, "index.html")

    @app.route("/camera")
    def camera():
        return send_from_directory(WEBSITE_DIR, "camera.html")

    @app.route("/<path:filename>")
    def static_files(filename):
        return send_from_directory(WEBSITE_DIR, filename)

    # ── Health check ───────────────────────────────────────────────

    @app.route("/api/health")
    def health():
        return jsonify({
            "status": "ok",
            "model": os.path.exists(ONNX_WEB),
            "classes": len(classes),
            "version": "1.0"
        })

    # ── Model info ─────────────────────────────────────────────────

    @app.route("/api/model_info")
    def model_info():
        ckpt_path = os.path.join(CKPT_DIR, "best_static.pt")
        info      = {"classes": classes, "num_classes": len(classes)}
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            info["val_acc"]  = round(float(ckpt.get("val_acc", 0)) * 100, 2)
            info["epoch"]    = ckpt.get("epoch", "?")
        return jsonify(info)

    # ── Load PyTorch model once at startup ───────────────────────────
    _pt_model = [None]  # mutable container for lazy load

    def get_model():
        if _pt_model[0] is not None:
            return _pt_model[0]
        ckpt_path = os.path.join(CKPT_DIR, "best_static.pt")
        if not os.path.exists(ckpt_path):
            return None
        try:
            from models.static_mlp import StaticGestureNet
            ckpt  = torch.load(ckpt_path, map_location="cpu")
            model = StaticGestureNet(num_classes=len(classes))
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            _pt_model[0] = model
            print(f"[predict] PyTorch model loaded ({len(classes)} classes)")
            return model
        except Exception as e:
            print(f"[predict] Model load error: {e}")
            return None

    # ── /api/predict — receives landmarks, returns class probs ────────
    @app.route("/api/predict", methods=["POST"])
    def predict():
        try:
            data = request.get_json(force=True)
            lm   = np.array(data["landmarks"], dtype=np.float32)  # (21, 3)

            # Normalize — same as preprocess.py
            lm  -= lm[0]
            scale = float(np.max(np.linalg.norm(lm, axis=1))) + 1e-6
            lm  /= scale
            feat = torch.from_numpy(lm.flatten()).unsqueeze(0)  # (1, 63)

            model = get_model()
            if model is None:
                return jsonify({"error": "Model not loaded"}), 503

            with torch.no_grad():
                logits = model(feat)[0].numpy()

            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            top   = int(probs.argmax())

            return jsonify({
                "sign":       classes[top],
                "confidence": float(probs[top]),
                "probs":      probs.tolist()
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 400

    return app


# ──────────────────────────────────────────────────────────────────
# Step 4 — Open browser after server starts
# ──────────────────────────────────────────────────────────────────

def open_browser(port: int, delay: float = 1.5):
    """Wait for server to start, then open camera page in browser."""
    def _open():
        time.sleep(delay)
        url = f"http://localhost:{port}/camera"
        print(f"\n[run_app] Opening browser → {url}")
        webbrowser.open(url)
    threading.Thread(target=_open, daemon=True).start()


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Launch SignSense web app with live camera inference"
    )
    parser.add_argument("--config", default="configs/static.yaml")
    parser.add_argument("--port",   type=int, default=5000)
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't open browser automatically")
    args = parser.parse_args()

    # Load config
    cfg_path = os.path.join(ROOT, args.config)
    if not os.path.exists(cfg_path):
        print(f"[run_app] Config not found: {cfg_path}")
        sys.exit(1)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    print("\n" + "═"*52)
    print("  SIGNSENSE  —  Web App Launcher")
    print("═"*52)

    # Step 1: Export ONNX
    onnx_ok = ensure_onnx(cfg)

    # Step 2: Copy to website/
    if onnx_ok:
        copy_model_to_website()
    else:
        print("[run_app] WARNING: No model — running in landmark-only mode")

    # Step 3: Create Flask app
    app = create_app(cfg)

    # Step 4: Auto-open browser
    if not args.no_browser:
        open_browser(args.port)

    # Step 5: Start server
    url = f"http://localhost:{args.port}"
    print(f"\n[run_app] Server started")
    print(f"  Home   → {url}/")
    print(f"  Camera → {url}/camera")
    print(f"  API    → {url}/api/health")
    print(f"\n  Press Ctrl+C to stop\n")

    app.run(
        host="0.0.0.0",
        port=args.port,
        debug=False,
        use_reloader=False
    )


if __name__ == "__main__":
    main()
