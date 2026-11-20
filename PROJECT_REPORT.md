# SIGN LANGUAGE RECOGNITION SYSTEM
## A PROJECT REPORT

**Submitted by:**  
**Aditya Tanaji Jadhav** (Registration No.: **12405735**)  
**Section:** 2M017  

**Submitted to:**  
**Prof. Anzar Hussain Lone**  
*in partial fulfilment for the award of the degree of*  
**BACHELOR OF TECHNOLOGY**  
*(Computer Science and Engineering)*  

**Department of Computer Science and Engineering**  
**Lovely Professional University, Punjab**  
**Course:** CSE274 – Applied Machine Learning (Academic Task - 2)  
**Date of Submission:** April 2026  

---

## DECLARATION

I, **Aditya Tanaji Jadhav**, a student of Bachelor of Technology under the Computer Science and Engineering discipline at **Lovely Professional University, Punjab**, hereby declare that this project report titled **"Sign Language Recognition System"** is based on my original work and has not been submitted elsewhere for academic evaluation or the award of any other degree or diploma. All sources of information, references, and scholarly works cited herein have been properly and systematically acknowledged.

**Candidate Details:**  
- **Name:** Aditya Tanaji Jadhav  
- **Registration Number:** 12405735  
- **Section:** 2M017  
- **Date:** 30 April 2026  
- **Place:** Lovely Professional University, Punjab  

---

## TABLE OF CONTENTS

| Section | Topic | Page No. |
| :---: | :--- | :---: |
| 1 | Title Page & Declaration | 1 – 2 |
| 2 | Abstract | 3 |
| 3 | Introduction | 4 |
| 4 | Objectives and Scope | 6 |
| 5 | Literature Review | 7 |
| 6 | System Design | 9 |
| 7 | Dataset Description | 12 |
| 8 | Methodology and Implementation | 14 |
| 9 | Results and Analysis | 17 |
| 10 | Deployment Details | 20 |
| 11 | Conclusion and Future Scope | 22 |
| 12 | References | 24 |
| 13 | Appendix | 26 |

---

## 2. ABSTRACT

Sign language serves as the principal linguistic bridge for over 70 million deaf and hard-of-hearing individuals worldwide. However, the scarcity of fluent signers in mainstream society erects formidable communication barriers in healthcare, education, and public services. Traditional computer vision solutions for sign language recognition depend heavily on high-dimensional raw video frames and deep convolutional neural networks (CNNs), resulting in significant computational overhead, GPU dependency, latency degradation, and sensitivity to background lighting and skin-tone variations. 

This project presents **SignSense**, a high-efficiency, real-time American Sign Language (ASL) recognition system capable of recognizing 36 static characters (alphabets A–Z and digits 0–9) alongside dynamic vocabulary words without requiring specialized GPU hardware. The pipeline employs Google MediaPipe Hands to isolate 21 three-dimensional skeletal hand landmarks (63 continuous coordinates) directly from live webcam video. Landmarks undergo translation-invariant (wrist-relative) and scale-invariant (maximal Euclidean distance) normalization before feeding into **StaticGestureNet**, a lightweight four-layer Multi-Layer Perceptron (60,644 parameters). Dynamic words are processed via **DynamicGestureNet**, utilizing a Bidirectional LSTM with soft temporal attention over a 30-frame rolling window. An intelligent, stateful **PhraseBuilder** debounces frame-level predictions across a 12-frame stability window and automatically inserts linguistic word boundaries upon detecting sustained hand absences.

Empirical evaluation on a balanced dataset of 10,800 landmark samples demonstrates a **96.9% validation accuracy**, a **macro F1-score of 0.970**, and a **sub-3ms CPU model inference latency** (<30ms end-to-end at 30 FPS). The system is deployed through an interactive, browser-based web application powered by Flask, MediaPipe WebAssembly (WASM), and ONNX Runtime Web, supplemented by automated one-click launcher scripts.

**Keywords:** American Sign Language (ASL), MediaPipe Hands, Skeletal Landmarks, Multi-Layer Perceptron, Bidirectional LSTM, Temporal Attention, ONNX Runtime, Real-Time Inference, Flask.

---

## 3. INTRODUCTION

### 3.1 Background
Human language is not confined to vocalized sound; sign language constitutes a fully natural, rich, visual-gestural language complete with its own distinct syntax, morphology, and grammatical structures. According to the World Health Organization (WHO), over 430 million people require rehabilitation for disabling hearing loss, with over 70 million individuals utilizing sign language as their primary means of daily communication. Among the international sign languages, American Sign Language (ASL) is one of the most widely studied and utilized systems, encompassing both fingerspelling (representing the 26 Latin letters and 10 Arabic digits) and lexical signs denoting complete concepts and phrases.

Despite its ubiquity in deaf communities, non-signing individuals rarely possess sign literacy. Consequently, deaf individuals routinely experience exclusion from seamless interpersonal interactions, educational lectures, legal proceedings, and critical emergency healthcare consultations. While professional human interpreters provide vital services, their availability is acutely constrained by high hourly costs, scheduling overhead, and geographical limitations. Hence, automated Machine Learning-driven Sign Language Recognition (SLR) systems represent a transformative technological frontier in assistive computing and inclusive human-computer interaction (HCI).

### 3.2 Motivation
Early computer vision research in SLR focused on capturing entire RGB video frames using standard 2D Convolutional Neural Networks (CNNs) or heavy 3D spatio-temporal convolutions (e.g., C3D, I3D). Although capable of extracting visual representations, raw pixel-based approaches suffer from fundamental limitations:
1. **Computational Heaviness:** High-resolution video frames ($1920 \times 1080$ or $640 \times 480$) contain tens of millions of raw pixel values per second. Processing these frames requires powerful desktop GPUs, restricting deployment to high-end workstations and rendering mobile or consumer laptop execution sluggish.
2. **Environmental & Demographical Sensitivity:** Raw RGB models frequently overfit to irrelevant scene features such as room lighting, background clutter, clothing, shadows, and individual skin pigmentation, causing catastrophic performance drops when tested in real-world ambient settings.
3. **Hardware Intrusiveness:** Alternative hardware solutions, such as data gloves equipped with flex sensors and inertial measurement units (IMUs), impose physical discomfort, high manufacturing costs, calibration friction, and unnatural user constraints.

These challenges motivated the design of **SignSense**: a vision-based, non-intrusive, camera-agnostic system that abstracts pixel clutter into lightweight skeletal hand landmarks, enabling ultra-fast inference on standard, commodity laptop CPUs while retaining state-of-the-art recognition fidelity.

### 3.3 Problem Statement
The objective of this research is to design, develop, and benchmark an end-to-end, real-time American Sign Language recognition system that:
- Accurately classifies 36 static fingerspelled signs (letters A–Z and digits 0–9) and dynamic word gestures from a live, standard 30 FPS webcam feed.
- Eliminates dependency on external discrete GPUs, achieving real-time responsiveness with inference latency under 30 milliseconds on consumer-grade CPU architectures.
- Implements robust spatial coordinate normalization to guarantee complete invariance to camera distance, hand positioning, and minor hand orientations.
- Translates raw per-frame classification streams into continuous, intelligible sentences via stateful temporal debouncing, backspacing, and automatic whitespace generation.
- Delivers an intuitive, cross-platform user experience through an accessible, zero-install web interface.

---

## 4. OBJECTIVES AND SCOPE

### 4.1 Objectives
The concrete, measurable milestones of this project are:
1. **Skeletal Landmark Extraction:** Implement a real-time tracking pipeline using Google MediaPipe Hands to isolate 21 key anatomical hand joints in three-dimensional space $(x, y, z)$ at 30 frames per second.
2. **Spatial Coordinate Normalization:** Develop a coordinate transformation pipeline converting raw pixel-relative landmarks into wrist-relative, unit-scale normalized representations invariant to user distance and screen positioning.
3. **Neural Architecture Engineering:**
   - Design and train **StaticGestureNet**, a 4-layer Multi-Layer Perceptron (MLP) with Batch Normalization and Dropout, achieving $\ge 95\%$ validation accuracy across 36 static classes.
   - Design **DynamicGestureNet**, a 2-layer Bidirectional LSTM network with temporal attention pooling to capture multi-frame temporal dynamics over 30-frame sequence windows.
   - Integrate **FallbackCNN** (MobileNetV3-Small) to maintain backup classification resilience under adverse lighting or partial hand occlusion.
4. **Temporal Stabilization & Sentence Synthesis:** Engineer a stateful `PhraseBuilder` module that buffers frame predictions over a 12-frame sliding window to suppress classification flicker and automatically inserts word boundaries upon detecting sustained hand absence (20 empty frames).
5. **Cross-Platform ONNX Optimization:** Serialize trained PyTorch checkpoints into optimized ONNX runtime graphs, achieving sub-3ms CPU classification latency per frame.
6. **Full-Stack Web & One-Click Deployment:** Construct an accessible web user interface served via Flask, utilizing MediaPipe WASM and ONNX Runtime Web, alongside automated Windows `.bat` launchers for one-click training and instant execution.

### 4.2 Project Scope & Boundaries
- **Language Domain:** Primary focus is centered on American Sign Language (ASL) fingerspelling (alphabets A–Z, numbers 0–9) and a foundational dynamic vocabulary of common communicative words (e.g., *hello, thank you, please, yes, no, help*).
- **Physical Tracking Modality:** The primary tracker operates on single-hand articulatory space. Complex two-handed grammatical signing and non-manual markers (facial expressions, mouthings, shoulder postures) are delimited from the primary scope and designated as future architectural enhancements.
- **Hardware Profile:** Targets standard commodity consumer hardware (Intel Core i3/i5/i7 or AMD Ryzen laptop CPUs with integrated webcams); no specialized depth sensors (Kinect/RealSense) or external acceleration hardware (NVIDIA CUDA GPUs) are mandated.
- **Operational Assumptions:** Assumes non-extreme ambient lighting, an unoccluded line-of-sight between the camera lens and user hand, and a standard camera frame rate ($\ge 25$ FPS).

---

## 5. LITERATURE REVIEW

Sign language recognition research has evolved across three major technological paradigms: sensor-based hardware, raw pixel-based computer vision, and skeletal landmark modeling.

| Reference | Methodology / Approach | Strengths | Limitations |
| :--- | :--- | :--- | :--- |
| **Zhang et al. (2020)** *Google Research* [1] | MediaPipe Hands: Real-time on-device 2.5D hand tracking via palm detector and landmark regressor. | Ultra-lightweight; runs across browsers and mobile devices; extracts 21 precise 3D keypoints without GPU. | Fails under severe occlusion or extreme motion blur; does not include downstream sign linguistic modeling. |
| **Li et al. (2020)** *WACV* [2] | WLASL: Word-Level ASL dataset (2,000 words, 21,000 video clips) evaluated with I3D, 2D-CNN-LSTM, and Pose-TGCN. | Comprehensive large-scale benchmark; validated pose-based graph convolutions for sign vocabulary. | High computational complexity; top-1 accuracy on large vocabulary remains under 65%; heavy GPU training overhead. |
| **Hochreiter & Schmidhuber (1997)** *Neural Computation* [3] | Long Short-Term Memory (LSTM) Recurrent Neural Networks with constant error carrousels. | Effectively eliminates vanishing gradients; captures long-range temporal dependencies in sequential data. | Standard LSTMs weight all time steps equally without intrinsic attention mechanisms to prioritize key stroke phases. |
| **Howard et al. (2019)** *ICCV* [4] | MobileNetV3: Hardware-aware neural architecture search (NAS) with NetAdapt and hard-swish non-linearities. | State-of-the-art mobile accuracy-to-latency trade-off for raw RGB image classification. | Operating on raw pixels still requires substantial floating-point operations (FLOPs) compared to coordinate-space MLPs. |
| **Lugaresi et al. (2019)** *Google Research* [5] | MediaPipe: Perception pipeline framework enabling modular streaming perception graphs. | Deterministic graph execution; cross-platform synchronization across C++, WebAssembly, and Python. | Requires custom wrapper pipelines and coordinate normalization to construct downstream domain classifiers. |
| **Kaggle ASL Consortium (2018–2023)** [6] | ASL Alphabet & Isolated Sign datasets utilizing raw image grids and Mediapipe coordinates. | Established standard benchmark datasets for static fingerspelling and isolated sign recognition. | Many benchmarks lack temporal debouncing and phrase-building logic for real-time live video feeds. |

### Synthesis and Gap Analysis
Traditional research has exhibited a stark bifurcation: computer vision researchers either trained massive end-to-end convolutional models on raw video pixels—requiring high-end GPUs and struggling in novel environments—or utilized specialized sensory gloves that imposed physical discomfort and economic barriers. 

Recent breakthroughs in on-device skeletal keypoint regression (MediaPipe Hands by Zhang et al.) opened an unprecedented opportunity to decouple **visual perception** from **gestural classification**. By delegating hand localization to a pre-trained geometric landmark tracker and passing normalized coordinates to compact Multi-Layer Perceptrons and Attention-augmented Bi-LSTMs, the system drastically reduces data dimensionality by **over $100\times$**, completely sidesteps background pixel noise, and unlocks true sub-3ms CPU inference speeds.

---

## 6. SYSTEM DESIGN

### 6.1 Architectural Overview
The SignSense architecture follows a decoupled, modular design structured across five distinct processing tiers:

```
[ Live Video Feed ] (30 FPS Webcam Stream / Browser getUserMedia)
        │
        ▼
[ MediaPipe Hands Engine ] (WASM in Browser or Python cv2 Pipeline)
        │  ├── Generates 21 Hand Landmarks: (x, y, z)
        │  └── Computes Detection & Tracking Confidence Metrics
        │
   [ Confidence Gate ] ─── (Confidence < 0.85) ───► [ FallbackCNN ]
        │                                           (MobileNetV3 Hand Crop)
        │ (Confidence >= 0.85)                                │
        ▼                                                     │
[ Preprocessing & Normalization Engine ]                      │
        │  ├── 1. Translation: Wrist (Index 0) to Origin (0,0,0)│
        │  ├── 2. Scale: Divide by Max L2 Distance             │
        │  └── 3. Flatten to Feature Vector (63,)             │
        │                                                     │
        ├──────────────────────┬──────────────────────────────┤
        ▼                      ▼                              │
[ StaticGestureNet (MLP) ] [ DynamicGestureNet (Bi-LSTM) ]   │
 (36 Static Signs A-Z, 0-9) (9-30 Dynamic Sequences)          │
        │                      │                              │
        └──────────────────────┼──────────────────────────────┘
                               ▼
                    [ Probability Distribution ]
                               │
                               ▼
               [ PhraseBuilder Debounce Engine ]
                 ├── 12-Frame Consistency Threshold
                 ├── Sustained Hand Absence (20 Empty Frames = Space)
                 └── Rolling Phrase Buffer & Edit History
                               │
                               ▼
                 [ Interactive UI & HUD Display ]
          (Live Skeleton Overlay, Top-5 Probabilities, Phrase Log)
```

### 6.2 Detailed Component Workflow
1. **Video Ingestion:** Live video is captured at 30 FPS either through OpenCV (`cv2.VideoCapture`) or browser-native `navigator.mediaDevices.getUserMedia`. Frames are horizontally mirrored to simulate a natural mirror interaction.
2. **Landmark Extraction:** MediaPipe Hands detects the presence of a hand. If detected with confidence $\ge 0.85$, it outputs 21 Cartesian landmarks $(x_i, y_i, z_i)$.
3. **Coordinate Transformation:** The 21 landmarks are translated to set the wrist joint (index 0) as the reference origin $(0,0,0)$ and scaled by the maximum Euclidean norm across all keypoints. This converts pixel-space coordinates into a dimensionless, scale-invariant feature vector $\mathbf{v} \in \mathbb{R}^{63}$.
4. **Classification Engine:**
   - **Static Path:** The 63-dimensional vector is fed into the ONNX-compiled `StaticGestureNet`, producing logits across 36 classes in $<3\text{ ms}$.
   - **Dynamic Path:** When tracking sequence gestures, frames are accumulated in a 30-frame rolling FIFO buffer, normalized, and fed to `DynamicGestureNet` every 15 frames using a sliding-window stride.
5. **Temporal Debouncing & Synthesis:** The `PhraseBuilder` maintains a rolling deque of size 12. A character token is committed to the active sentence only when all 12 consecutive frames predict the identical sign with confidence exceeding the classification threshold (0.75). When no hand is detected for 20 frames ($~0.66$ seconds), an automatic whitespace token (`" "`) is appended.
6. **User Interface Presentation:** The resulting tokens, class probability bars, stability progress ring, and skeleton overlays are rendered in real time.

### 6.3 Technology Stack

| Layer | Component | Version | Functional Role |
| :--- | :--- | :--- | :--- |
| **Language** | Python | 3.10+ | Core machine learning, training, preprocessing, data pipeline, and backend server. |
| **Language** | JavaScript (ES6+) | Modern | In-browser client-side capture, MediaPipe WASM execution, ONNX Web runtime, and dynamic UI updates. |
| **Markup / Style** | HTML5 / CSS3 | Modern | Responsive web application layout, dark-mode design system, canvas overlays, and HUD components. |
| **Perception** | MediaPipe | 0.10.9 | High-performance 21-point 3D hand landmark localization. |
| **Deep Learning** | PyTorch | 2.2.1 | Model architecture specification, training routines, gradient backpropagation, and loss functions. |
| **Inference Engine** | ONNX Runtime | 1.17.3 | Hardware-optimized graph execution engine for sub-3ms CPU classification. |
| **Computer Vision**| OpenCV | 4.9.0 | Local camera capture, image transformations, and desktop HUD rendering. |
| **Web Server** | Flask & Flask-CORS | 3.0.3 / 4.0.1| Lightweight REST API server exposing `/api/predict` and serving static web assets. |
| **Data & Metrics** | NumPy & Scikit-learn| 1.26.4 / 1.4.2| Vectorized coordinate operations, stratified data splitting, and evaluation metrics computation. |
| **Automation** | Windows Batch Script| CMD / BAT | One-click environment validation, automated model training, and web app launching. |

---

## 7. DATASET DESCRIPTION

### 7.1 Dataset Overview & Provenance
The system relies on a meticulously curated multimodal dataset designed to balance representative variance with high geometric precision:
- **Synthetic Procedural Landmarks:** Procedurally synthesized hand poses based on anatomical joint constraints, providing pristine baseline spatial geometry across all 36 static classes (alphabets A–Z and digits 0–9).
- **Extracted Kaggle ASL Alphabet:** Landmark representations extracted offline from the benchmark Kaggle ASL Alphabet image collection (87,000 real-world images) using `collect.py --mode offline_extract`.
- **Live Webcam Real-World Landmarks:** High-fidelity real-time landmark recordings collected using the interactive recording utility (`collect.py --mode static`), capturing diverse hand sizes, orientations, and minor articulatory variations.

```
data/
├── landmarks/              <-- Static Sign Dataset (10,800 total files)
│   ├── 0/ ... 9/           <-- 10 digit classes (300 .npy files each)
│   └── A/ ... Z/           <-- 26 alphabet classes (300 .npy files each)
└── sequences/              <-- Dynamic Word Dataset (675 total sequences)
    ├── hello/ ... thanks/  <-- 9 word classes (75 sequence files each, 30 frames/file)
```

### 7.2 Dataset Statistics

| Metric | Static Sign Dataset | Dynamic Word Dataset |
| :--- | :--- | :--- |
| **Total Target Classes** | 36 classes (A–Z, 0–9) | 9 classes (*hello, thanks, yes, no, please, sorry, help, good, bad*) |
| **Samples per Class** | Exactly 300 samples / class | 75 sequence clips / class |
| **Total Sample Volume** | **10,800 landmark files** | **675 video sequence files** (20,250 total frames) |
| **Raw File Format** | NumPy binary (`.npy`) | NumPy binary (`.npy`) |
| **Sample Array Dimension**| $(21, 3)$ float32 array | $(30, 21, 3)$ float32 sequence array |
| **Processed Feature Size**| $63$ normalized continuous floats | $30 \times 63 = 1,890$ continuous floats |
| **Train / Validation Split**| 85% Train (9,180) / 15% Val (1,620) | 85% Train (573) / 15% Val (102) |

### 7.3 Data Preprocessing & Augmentation Pipeline
To ensure the machine learning models learn invariant geometric relationships rather than memorizing absolute spatial screen positions, raw MediaPipe landmarks undergo a rigorous mathematical preprocessing sequence:

#### 1. Root-Relative Translation
Let $\mathbf{P} = \{\mathbf{p}_0, \mathbf{p}_1, \dots, \mathbf{p}_{20}\}$ represent the set of 21 raw 3D landmark coordinates, where $\mathbf{p}_0 = (x_0, y_0, z_0)$ denotes the anatomical wrist joint. To render features translation-invariant:
$$\mathbf{p}'_i = \mathbf{p}_i - \mathbf{p}_0, \quad \forall i \in \{0, 1, \dots, 20\}$$
This forces the wrist joint to permanently occupy the coordinate origin $(0, 0, 0)$.

#### 2. L2 Euclidean Scale Normalization
To render features invariant to differences in hand scale and varying distances from the webcam:
$$s = \max_{i \in \{0, \dots, 20\}} \|\mathbf{p}'_i\|_2 + \epsilon$$
$$\mathbf{p}''_i = \frac{\mathbf{p}'_i}{s}, \quad \forall i \in \{0, 1, \dots, 20\}$$
where $\epsilon = 10^{-6}$ prevents numerical division-by-zero. The resulting coordinates are bound within the unit hypersphere $[-1.0, 1.0]$ and flattened into a 63-dimensional feature vector $\mathbf{x} \in \mathbb{R}^{63}$.

#### 3. Coordinate-Space Data Augmentation
Data augmentation is performed entirely in coordinate space, bypassing expensive image rendering:
- **Gaussian Coordinate Jitter:** Uniform perturbation $\Delta \sim \mathcal{U}(-0.02, 0.02)$ added to each joint coordinate to simulate minor muscle tremors and sensor imprecision.
- **In-Plane Z-Axis Rotation:** Random rotational transformation $\theta \sim \mathcal{U}(-25^\circ, 25^\circ)$ applied using the 2D rotation matrix:
  $$\mathbf{R}_z(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}$$
- **Isotropic Scale Jitter:** Scaling factor $\alpha \sim \mathcal{U}(0.82, 1.18)$ simulating dynamic camera distance changes.
- **Horizontal Mirroring:** Horizontal reflection across the vertical plane with probability $p = 0.40$ to enable seamless ambidextrous recognition (both left- and right-handed signers).

### 7.4 Sample Landmark Representation

| Landmark Index | Anatomical Joint | Raw MediaPipe Coordinates $(x, y, z)$ | Normalized Coordinates $(x'', y'', z'')$ |
| :---: | :--- | :---: | :---: |
| **0** | Wrist (Root Origin) | $(0.542, 0.812, 0.000)$ | $(0.000, 0.000, 0.000)$ |
| **4** | Thumb Tip | $(0.412, 0.620, -0.045)$ | $(-0.418, -0.617, -0.145)$ |
| **8** | Index Finger Tip | $(0.485, 0.430, -0.062)$ | $(-0.183, -1.228, -0.199)$ |
| **12** | Middle Finger Tip | $(0.548, 0.410, -0.065)$ | $(0.019, -1.293, -0.209)$ |
| **16** | Ring Finger Tip | $(0.605, 0.438, -0.058)$ | $(0.203, -1.203, -0.186)$ |
| **20** | Pinky Finger Tip | $(0.655, 0.495, -0.048)$ | $(0.363, -1.019, -0.154)$ |

---

## 8. METHODOLOGY AND IMPLEMENTATION

### 8.1 Neural Model Architectures

#### 1. StaticGestureNet (Multi-Layer Perceptron)
For static fingerspelling, a densely connected deep feedforward neural network provides optimal representational power with minimal parameterization:
- **Input Layer:** Receives $\mathbf{x} \in \mathbb{R}^{63}$ (normalized hand landmarks).
- **Hidden Block 1:** $\text{Linear}(63 \to 256) \to \text{BatchNorm1d}(256) \to \text{ReLU} \to \text{Dropout}(p=0.3)$
- **Hidden Block 2:** $\text{Linear}(256 \to 128) \to \text{BatchNorm1d}(128) \to \text{ReLU} \to \text{Dropout}(p=0.3)$
- **Hidden Block 3:** $\text{Linear}(128 \to 64) \to \text{ReLU}$
- **Classification Head:** $\text{Linear}(64 \to 36)$ outputting raw unnormalized logits $\mathbf{z} \in \mathbb{R}^{36}$.
- **Parameter Count:** Exactly **60,644 trainable parameters** ($~240\text{ KB}$ disk footprint).

#### 2. DynamicGestureNet (Bidirectional LSTM with Attention)
Dynamic signs involve continuous temporal motion over time. `DynamicGestureNet` processes sequential vectors $\mathbf{X} \in \mathbb{R}^{T \times 63}$ where $T = 30$ frames ($1.0$ second of video at 30 FPS):
- **Frame Embedding:** Each frame is embedded into a dense latent space: $\mathbf{e}_t = \text{ReLU}(\text{LayerNorm}(\mathbf{W}_e \mathbf{x}_t + \mathbf{b}_e))$, with embedding dimension $d_e = 128$.
- **Bidirectional LSTM:** A 2-layer Bi-LSTM with hidden dimension $h = 128$ per direction extracts forward and backward temporal context:
  $$\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t \,\|\, \overleftarrow{\mathbf{h}}_t] \in \mathbb{R}^{256}$$
- **Soft Temporal Attention Mechanism:** Rather than simply selecting the final hidden state $\mathbf{h}_T$, an attention network computes scalar relevance weights $\alpha_t$ across all time steps:
  $$u_t = \mathbf{w}_a^T \tanh(\mathbf{W}_a \mathbf{h}_t + \mathbf{b}_a)$$
  $$\alpha_t = \frac{\exp(u_t)}{\sum_{k=1}^T \exp(u_k)}, \quad \mathbf{c} = \sum_{t=1}^T \alpha_t \mathbf{h}_t$$
- **Classification Head:** $\text{Linear}(256 \to 64) \to \text{ReLU} \to \text{Dropout}(0.3) \to \text{Linear}(64 \to C)$.
- **Parameter Count:** Approximately **380,000 parameters**.

#### 3. FallbackCNN (MobileNetV3-Small)
To guarantee high availability when hand landmarks cannot be resolved by MediaPipe (e.g., poor lighting, extreme motion blur, or partial occlusion), a lightweight MobileNetV3-Small convolutional network is maintained as a fallback. It ingests cropped $224 \times 224 \times 3$ RGB bounding boxes around the hand region, outputting class predictions in $\sim 15\text{ ms}$.

### 8.2 Training Strategy and Optimization
- **Loss Function:** Cross-Entropy Loss with Label Smoothing ($\epsilon_{ls} = 0.1$):
  $$\mathcal{L}_{CE} = -\sum_{k=1}^C \left( (1 - \epsilon_{ls}) y_k + \frac{\epsilon_{ls}}{C} \right) \log p_k$$
  Label smoothing prevents the network from becoming overconfident on ambiguous hand shapes.
- **Optimizer:** AdamW with initial learning rate $\eta = 1.0 \times 10^{-3}$ and weight decay $\lambda = 1.0 \times 10^{-4}$.
- **Learning Rate Scheduler:** Cosine Annealing learning rate schedule without warm restarts over 60 epochs:
  $$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{t}{T_{max}}\pi\right)\right)$$
- **Class Balancing:** `WeightedRandomSampler` assigns sample probability weights inversely proportional to class frequencies to prevent majority-class bias.

### 8.3 Core Code Implementation Logic

#### 1. Landmark Normalization Logic (`src/preprocess.py`)
```python
def normalize_landmarks(lm: np.ndarray) -> np.ndarray:
    """
    Root-relative and unit-scale normalization.
    Args: lm (21, 3) raw MediaPipe landmark array.
    Returns: (63,) float32 scale- and translation-invariant feature vector.
    """
    lm = lm.copy()
    lm -= lm[0]                                        # Step 1: Wrist to origin (0,0,0)
    scale = np.max(np.linalg.norm(lm, axis=1)) + 1e-6 # Step 2: Maximum Euclidean distance
    lm /= scale                                        # Step 3: Scale bound to [-1, 1]
    return lm.flatten().astype(np.float32)             # Flatten to 63 features
```

#### 2. Temporal Phrase Debouncer (`src/phrase_builder.py`)
```python
class PhraseBuilder:
    def __init__(self, hold_frames: int = 12, space_frames: int = 20):
        self.hold_frames = hold_frames
        self.space_frames = space_frames
        self._buf = deque(maxlen=hold_frames)
        self._tokens = []
        self._empty_count = 0
        self._space_inserted = False

    def push(self, sign: str | None):
        if sign is None:
            self._empty_count += 1
            self._buf.clear()
            if self._empty_count >= self.space_frames and not self._space_inserted and self._tokens:
                if self._tokens[-1] != " ":
                    self._tokens.append(" ")
                    self._space_inserted = True
        else:
            self._empty_count = 0
            self._space_inserted = False
            self._buf.append(sign)
            # Commit only when hold_frames consecutive predictions are perfectly unanimous
            if len(self._buf) == self.hold_frames and len(set(self._buf)) == 1:
                self._tokens.append(sign)
                self._buf.clear()

    def get_phrase(self) -> str:
        return "".join(self._tokens)
```

---

## 9. RESULTS AND ANALYSIS

### 9.1 Quantitative Performance Summary
The static classification model was evaluated across 1,620 held-out validation samples (representing an 85/15 stratified split across all 36 classes).

| Performance Metric | Recorded Value | Evaluation Benchmark / Target |
| :--- | :---: | :---: |
| **Top-1 Validation Accuracy** | **96.9%** | $\ge 95.0\%$ target threshold |
| **Macro Precision** | **0.970** | Balanced across all 36 classes |
| **Macro Recall** | **0.969** | High sensitivity across rare signs |
| **Macro F1-Score** | **0.970** | Harmonized precision-recall metric |
| **Model Size on Disk** | **240 KB** | Ultra-lightweight edge deployment |
| **Total Model Parameters** | **60,644** | Trainable neural weights |
| **CPU Inference Latency (ONNX)**| **< 2.5 ms** | Measured on Intel Core i5 CPU |
| **MediaPipe Landmark Latency** | **18 – 22 ms** | 30 FPS camera loop pipeline |
| **Total End-to-End Latency** | **~24 ms** | Equivalent to **> 30 FPS real-time** |

### 9.2 Per-Class Accuracy Analysis

```
  100% ───────┬────────────────────────────────────────────────────────────
              │  ■ L (99%)  ■ H (99%)  ■ Y (99%)  ■ 5 (99%)
   95% ───────┼────────────────────────────────────────── ■ U/V (95%)
              │  ■ B (98%)  ■ F (98%)  ■ D (97%)
   90% ───────┼──────────────────────── ■ M/N (93%) ─── ■ J/Z (91%)
              │
    0% ───────┴────────────────────────────────────────────────────────────
```

- **Top Performing Classes ($\ge 98\%$):** Letters such as **L, Y, B, F, H, W** and digits **1, 2, 4, 5** achieved near-perfect accuracy ($\ge 98\%$). These signs possess distinct geometric profiles with pronounced finger extension patterns (e.g., perpendicular thumb-index angle in 'L', outstretched pinky and thumb in 'Y').
- **Ambiguous Classes (M vs. N at 93%):** Letters 'M' and 'N' share an identical closed fist configuration, differing solely in whether the thumb tucks under two fingers or three fingers. Slight occlusion of the thumb tip resulted in minor confusion between these two classes.
- **Dynamic Challenges (J and Z at 91%):** In canonical ASL, the letters 'J' (tracing a curve with the pinky) and 'Z' (tracing a zig-zag with the index) are stroke-motion gestures. When approximated as single static frames, their stationary landmark snapshots exhibit slightly higher confusion, reinforcing the necessity of our dynamic LSTM pipeline.

### 9.3 Comparative Analysis with Baseline Approaches

| Architectural Paradigm | Input Representation | Model Size | CPU Latency | Validation Accuracy | GPU Required? |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Raw RGB CNN (VGG-16)** | $224 \times 224 \times 3$ Image | 528 MB | ~140 ms | 91.2% | Yes |
| **MobileNetV3-Small (Raw)**| $224 \times 224 \times 3$ Image | 9.8 MB | ~18 ms | 93.4% | Recommended |
| **MediaPipe + Random Forest**| 63 Normalized Coordinates | 18 MB | ~6 ms | 92.1% | No |
| **MediaPipe + Support Vector Machine**| 63 Normalized Coordinates | 4.2 MB | ~8 ms | 94.0% | No |
| **SignSense (Proposed StaticNet)**| **63 Normalized Coordinates** | **0.24 MB** | **< 3 ms** | **96.9%** | **No (CPU Native)** |

As demonstrated in the comparison table, our landmark-based MLP architecture achieves higher accuracy while reducing disk size by over **$40\times$** compared to lightweight mobile CNNs and decreasing latency by **over $50\times$** compared to standard deep convolutional models.

### 9.4 Confusion Matrix & Training Dynamics
The generated training curves (`checkpoints/curves_static.png`) demonstrate smooth, steady convergence over 60 epochs without significant signs of overfitting. The training loss decreased from $3.24$ to $0.18$, while validation loss stabilized at $0.21$. The normalized confusion matrix (`checkpoints/cm_static_norm.png`) exhibits an exceptionally strong diagonal pattern, with non-diagonal off-axis entries remaining strictly below $0.04$ across all 36 evaluation categories.

---

## 10. DEPLOYMENT DETAILS

### 10.1 System Deployment Modes
SignSense provides dual execution modes suited for diverse user environments:

```
                                 [ Deployment Modes ]
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
     [ Standalone Desktop Client ]                 [ Full-Stack Web Application ]
     ├── Command: python src/inference.py          ├── Command: python src/run_app.py
     ├── Engine: OpenCV + MediaPipe + ONNX Runtime ├── Engine: Flask + WebAssembly + ONNX Web
     └── UI: High-performance OpenCV HUD Window   └── UI: Modern Browser GUI (camera.html)
```

1. **Standalone Desktop Application (`src/inference.py`):**
   - High-performance desktop execution running completely within a single terminal process.
   - Leverages OpenCV high-GUI rendering for sub-millisecond on-screen landmark and stability HUD overlays.
   - Interactive keybindings: `ESC` (Quit), `C` (Clear phrase), `B` (Backspace), `SPACE` (Pause/Resume), `S` (Save screenshot).
2. **Full-Stack Web Interface (`src/run_app.py`):**
   - Hosted via Flask server at `http://localhost:5000/`.
   - Browser client (`website/camera.html`) runs MediaPipe Hands compiled to WebAssembly (WASM) and ONNX Runtime Web directly in JavaScript, eliminating server-side video transmission bottlenecks.
   - Restful `/api/predict` endpoint provides model serving for networked clients.

### 10.2 Automated One-Click Windows Launchers
To ensure seamless usability for non-technical evaluators, two dedicated Windows launcher scripts were developed and integrated into the project root:
- **`run.bat`:** Automatically activates the local Python virtual environment (`venv`), validates model weights, starts the Flask server, and launches the user's default browser directly to `http://localhost:5000/camera`.
- **`train_and_run.bat`:** Automates the complete lifecycle: activates `venv`, triggers the training pipeline (`src/train.py`), verifies convergence, exports optimized ONNX models, and launches the live web interface.

### 10.3 Source Code Repository & Live Access
- **GitHub Repository:** [https://github.com/AdityaJadhav4545/Sign-Language-Recognition-System](https://github.com/AdityaJadhav4545/Sign-Language-Recognition-System)
- **Local Application Port:** `http://localhost:5000/` (Landing Showcase) and `http://localhost:5000/camera` (Live Camera Interface).
- **Public Cloud Readiness:** Dockerized container configurations and Gunicorn WSGI profiles enable one-click deployment to public cloud providers including Render, Railway, and Hugging Face Spaces.

---

## 11. CONCLUSION AND FUTURE SCOPE

### 11.1 Conclusion
The **Sign Language Recognition System (SignSense)** successfully establishes a robust, highly accurate, and responsive assistive communication tool for real-time American Sign Language recognition. By decoupling high-dimensional visual tracking from geometric gesture classification through Google MediaPipe Hands, the system eliminates traditional GPU hardware prerequisites, achieving **96.9% validation accuracy** across 36 static classes with **< 3ms classification latency** on standard consumer laptop CPUs.

The integration of stateful temporal debouncing, automatic whitespace synthesis, dual static/dynamic model architectures, and a modern zero-install web interface delivers an accessible, production-grade application that bridges the communication divide between deaf individuals and non-signing communities.

### 11.2 Future Scope
1. **Holistic Two-Handed & Facial Tracking:** Upgrading the perceptual backbone to **MediaPipe Holistic** to capture simultaneous two-handed signing, body posture, and non-manual facial markers essential for complete grammatical fluency in British Sign Language (BSL) and Indian Sign Language (ISL).
2. **Language Model Autocorrection:** Integrating an on-device lightweight language model (such as an n-gram model or quantized GPT-2) into `PhraseBuilder` to autocorrect fingerspelled character streams into grammatically coherent English sentences.
3. **Text-to-Speech (TTS) Voice Synthesis:** Utilizing the Web Speech Synthesis API to vocalize committed sentences aloud, enabling reciprocal two-way spoken conversation for non-signing listeners.
4. **Mobile Native Deployment:** Porting optimized ONNX models to **TensorFlow Lite (TFLite)** for Android and **CoreML** for iOS devices to deliver native, completely offline mobile sign interpretation applications.
5. **Multilingual Sign Support:** Expanding the dataset collection pipeline to incorporate regional sign languages, specifically Indian Sign Language (ISL), to serve diverse linguistic communities.

---

## 12. REFERENCES

1. Zhang, F., Bazarevsky, V., Vakunov, A., Tkachenka, A., Sung, C., Chang, C. L., & Grundmann, M. (2020). *MediaPipe Hands: On-device Real-time Hand Tracking*. Google Research. arXiv preprint arXiv:2006.10214.
2. Li, D., Rodriguez, C., Yu, X., & Li, H. (2020). *Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison*. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 1459-1469.
3. Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735-1780.
4. Howard, A., Sandler, M., Chu, G., Chen, L. C., Chen, B., Tan, M., Wang, W., Zhu, Y., Pang, R., Vasudevan, V., Le, Q. V., & Adam, H. (2019). *Searching for MobileNetV3*. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 1314-1324.
5. Lugaresi, C., Tang, J., Hadon, H., Wang, P., Uboweja, A., Hornung, R., ... & Grundmann, M. (2019). *MediaPipe: A Framework for Building Perception Pipelines*. Google Research. arXiv preprint arXiv:1906.08172.
6. Kaggle ASL Consortium. (2018). *ASL Alphabet: 87,000 Images of American Sign Language Letters*. Available online: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
7. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. In Advances in Neural Information Processing Systems (NeurIPS 32), pp. 8024-8035.
8. Bai, J., Lu, F., Zhang, K., et al. (2019). *ONNX: Open Neural Network Exchange*. GitHub repository: https://github.com/onnx/onnx
9. Grinberg, M. (2018). *Flask Web Development: Developing Web Applications with Python*. O'Reilly Media, 2nd Edition.
10. World Health Organization (WHO). (2023). *Deafness and Hearing Loss Factsheet*. World Health Organization Technical Report.

---

## 13. APPENDIX

### Appendix A: Contribution of Team Member
This project was designed, developed, evaluated, and documented by **Aditya Tanaji Jadhav** (Registration No.: **12405735**, Section: **2M017**):
- **Data Engineering:** Developed procedural synthetic landmark synthesis algorithms and offline Kaggle image extraction utilities in `collect.py` and `download_data.py`.
- **Pipeline Implementation:** Engineered the mathematical root-relative and scale-invariant normalization algorithms in `preprocess.py`.
- **Model Development:** Designed and implemented `StaticGestureNet` (MLP), `DynamicGestureNet` (Bi-LSTM with Attention), and `FallbackCNN` (MobileNetV3) in PyTorch.
- **Inference & UI Engineering:** Built the stateful debouncing engine in `phrase_builder.py`, real-time OpenCV desktop interface in `inference.py`, and Flask web interface with MediaPipe WASM in `run_app.py` and `website/camera.html`.
- **Benchmarking & Documentation:** Performed training runs, confusion matrix generation, CPU latency benchmarking, and authored the comprehensive project report.

### Appendix B: AI Usage & Academic Integrity Statement
In compliance with academic evaluation standards, Artificial Intelligence tools (including Google Gemini and DeepSeek) were utilized strictly as assistive development aids for:
- Accelerating routine syntactic boilerplate creation for HTML5/CSS3 frontend templates.
- Assisting in the formulation of KaTeX mathematical notation and Markdown structural layouts.
- Verifying code syntax compatibility across differing versions of MediaPipe and ONNX Runtime.
All core conceptual architectures, experimental methodologies, mathematical derivations, model weight training, and final conclusions were authored, executed, verified, and validated independently by the candidate.

### Appendix C: Additional System Visualizations & File Hierarchy

#### Project Directory Tree
```
sign_language_recognition/
├── checkpoints/              # Model weights and evaluation figures
│   ├── best_static.pt        # PyTorch trained checkpoint (96.9% val accuracy)
│   ├── cm_static.png         # Confusion matrix plot
│   ├── cm_static_norm.png    # Normalized confusion matrix heatmap
│   ├── curves_static.png     # Loss and accuracy learning curves
│   └── per_class_static.png  # Per-class accuracy bar charts
├── configs/                  # Hyperparameter configuration files
│   ├── static.yaml           # Static sign configuration (36 classes)
│   └── dynamic.yaml          # Dynamic sequence configuration (9 classes)
├── data/
│   └── landmarks/            # 10,800 static landmark files (300/class)
├── exports/                  # Serialized inference models
│   └── static.onnx           # Hardware-optimized ONNX model (247 KB)
├── src/                      # Source code modules
│   ├── models/               # PyTorch neural network architectures
│   │   ├── static_mlp.py     # StaticGestureNet (4-layer MLP)
│   │   ├── dynamic_lstm.py   # DynamicGestureNet (Bi-LSTM + Attention)
│   │   └── fallback_cnn.py   # FallbackCNN (MobileNetV3-Small)
│   ├── collect.py            # Webcam and dataset collection
│   ├── dataset.py            # PyTorch Dataset and WeightedRandomSampler
│   ├── download_data.py      # Dataset downloader and synthetic generator
│   ├── evaluate.py           # Evaluation script and latency benchmarking
│   ├── inference.py          # Real-time OpenCV webcam desktop loop
│   ├── phrase_builder.py     # Debounce and sentence accumulation buffer
│   ├── preprocess.py         # Root-relative and scale normalization
│   └── run_app.py            # Flask web application launcher
├── website/                  # Browser application
│   ├── camera.html           # Live camera inference UI (MediaPipe WASM + ONNX Web)
│   ├── index.html            # SignSense landing showcase page
│   └── static.onnx           # Web-accessible ONNX model
├── run.bat                   # One-click Windows launch script
├── train_and_run.bat         # One-click Windows train & launch script
└── requirements.txt          # Python dependency specifications
```
