# ✅ Quick Setup — Run These Commands on Your PC

## Step 1 — Install the missing library
```powershell
pip install datasets
```
That's the only missing package. Everything else in requirements.txt is already working.

---

## Step 2 — Download real online data

Run any of these (all free, no account needed):

```powershell
# Option A — Everything at once (recommended)
python src/download_data.py --all

# Option B — Just the real letter images from HuggingFace (A-Y, 27K samples)
python src/download_data.py --source mnist

# Option C — Just synthetic digit landmarks (0-9, instant, no internet)
python src/download_data.py --source digits

# Option D — Fingerspelling pre-extracted landmarks
python src/download_data.py --source fingerspelling
```

Expected output after --all:
- Sign Language MNIST: 27,455 real ASL letter images downloaded and converted to landmarks
- Synthetic digits: 200 landmarks per digit class (0-9)
- All data saved to data/landmarks/{CLASS}/*.npy

---

## Step 3 — Check what downloaded
```powershell
python src/download_data.py --stats
python src/download_data.py --verify
```

---

## Step 4 — Train the model
```powershell
python src/train.py --config configs/static.yaml --model static
```
Training takes 3-5 minutes on CPU for 60 epochs.
Best checkpoint saved to checkpoints/best_static.pt
ONNX model exported to exports/static.onnx

---

## Step 5 — Evaluate
```powershell
python src/evaluate.py --config configs/static.yaml --model static
```
Outputs: accuracy, F1, confusion matrix PNG, per-class accuracy PNG, latency benchmark.

---

## Step 6 — Run real-time inference
```powershell
python src/inference.py --config configs/static.yaml
```

Controls in the webcam window:
- ESC    : Quit
- c      : Clear phrase
- b      : Backspace last sign
- SPACE  : Pause / Resume
- s      : Screenshot

---

## Troubleshooting

**"No hand detected" / low accuracy on digits 0-9**
Digits use synthetic landmarks for bootstrap. Add webcam samples:
```powershell
python src/collect.py --mode static --out data/landmarks --config configs/static.yaml
```
Press SPACE to start recording each class, ESC to skip.

**MediaPipe not found**
```powershell
pip install mediapipe
```

**Low FPS in inference window**
Normal on CPU. MediaPipe takes ~20ms, ONNX model adds <3ms.
Close other apps using the camera.

**J and Z not recognized**
J and Z are dynamic signs (motion-based). Use the dynamic model:
```powershell
python src/collect.py --mode dynamic --out data/sequences --config configs/dynamic.yaml
python src/train.py --config configs/dynamic.yaml --model dynamic
python src/inference.py --config configs/dynamic.yaml --model dynamic
```

**HuggingFace download is slow**
The Sign Language MNIST dataset is ~100MB. Let it run.
Or skip it and use only synthetic + webcam data:
```powershell
python src/download_data.py --source digits
python src/collect.py --mode static --out data/landmarks
```
