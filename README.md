# Sign Language Recognition System

Real-time ASL recognition — alphabets (A–Z), numbers (0–9), words, and phrases — using MediaPipe, PyTorch, and ONNX Runtime.

---

## Architecture

```
Webcam → MediaPipe Hands → Landmark Normalization → ONNX Classifier → PhraseBuilder → Display
                                                          ↑
                                               FallbackCNN (if landmarks fail)
```

**Two-model pipeline:**
| Task | Model | Input | Latency (CPU) |
|------|-------|-------|---------------|
| Static signs (A–Z, 0–9) | MLP | (63,) landmarks | <3ms |
| Dynamic words/phrases | Bi-LSTM | (30, 63) sequence | <10ms |
| Fallback (poor lighting) | MobileNetV3-Small | (3,224,224) image | ~15ms |

---

## Quickstart

### 1. Install dependencies
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Collect data
```bash
# Interactive webcam collection (SPACE to start each class, ESC to skip)
python src/collect.py --mode static --out data/landmarks --config configs/static.yaml

# Or extract from Kaggle ASL Alphabet dataset (offline, much faster)
# Download from: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
python src/collect.py --mode offline_extract --img_root data/raw --out data/landmarks
```

### 3. Check dataset balance
```bash
python src/preprocess.py --mode stats --src data/landmarks
```

### 4. Train
```bash
# Static model (A–Z + 0–9)
python src/train.py --config configs/static.yaml --model static

# Dynamic model (words/phrases) — requires sequence data
python src/train.py --config configs/dynamic.yaml --model dynamic
```

### 5. Evaluate
```bash
python src/evaluate.py --config configs/static.yaml --model static
```

### 6. Run real-time inference
```bash
python src/inference.py --config configs/static.yaml --model static
```

---

## Webcam Controls

| Key | Action |
|-----|--------|
| `ESC` | Quit |
| `c` | Clear phrase |
| `b` | Backspace last sign |
| `SPACE` | Pause / Resume |
| `s` | Save screenshot |

---

## Folder Structure

```
sign_language_recognition/
├── data/
│   ├── raw/              # original images (Kaggle dataset)
│   ├── landmarks/        # extracted .npy landmark arrays
│   └── augmented/        # pre-augmented landmark cache (optional)
├── src/
│   ├── collect.py        # Module 1 – data collection
│   ├── preprocess.py     # Module 2 – normalization & augmentation
│   ├── dataset.py        # PyTorch Dataset / DataLoader
│   ├── models/
│   │   ├── static_mlp.py   # MLP for static signs
│   │   ├── dynamic_lstm.py # Bi-LSTM for words/phrases
│   │   └── fallback_cnn.py # MobileNetV3 fallback
│   ├── train.py          # Module 4 – training pipeline
│   ├── inference.py      # Module 5 – real-time webcam loop
│   ├── phrase_builder.py # sign debounce + phrase accumulation
│   └── evaluate.py       # full evaluation + plots
├── configs/
│   ├── static.yaml       # hyperparams for static model
│   └── dynamic.yaml      # hyperparams for dynamic model
├── checkpoints/          # .pt model weights + confusion matrix plots
├── exports/              # .onnx models for deployment
└── requirements.txt
```

---

## Recommended Datasets

| Dataset | Signs | Size | Link |
|---------|-------|------|------|
| ASL Alphabet (Kaggle) | A–Z + space | 87,000 images | kaggle.com/grassknoted/asl-alphabet |
| WLASL | 2,000 words | 21,000 videos | github.com/dxli94/WLASL |
| ASL Signs (Kaggle 2023) | 250 signs | landmark CSVs | kaggle.com/competitions/asl-signs |

---

## Evaluation Targets

| Metric | Static | Dynamic |
|--------|--------|---------|
| Top-1 accuracy | >97% | >88% |
| Macro F1 | >0.95 | >0.85 |
| P95 latency | <5ms | <15ms |

---

## Scaling Roadmap

- **v1** Static A–Z + 0–9 MLP (this repo, 2-day project)
- **v2** 100-word LSTM (WLASL-100 dataset)
- **v3** Full phrase: Bi-LSTM + n-gram language model correction, two-hand support
- **v4** Production: ONNX → TensorRT / CoreML; MediaPipe Holistic for full BSL/ISL

---

## W&B Integration

Set your API key to enable experiment tracking:
```bash
export WANDB_API_KEY=your_key_here
python src/train.py --config configs/static.yaml
```

Metrics logged: `tr_loss`, `val_loss`, `tr_acc`, `val_acc`, `lr` per epoch.
