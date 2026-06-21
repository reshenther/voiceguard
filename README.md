# 🛡️ VoiceGuard

### Deepfake Audio Detection Using Multi-Feature Spectrogram Analysis

VoiceGuard is a deep learning system that detects whether a voice recording is **real human speech** or **AI-generated (deepfake) audio**, in real time, through a full-stack web application.

---

## 🎯 The Problem

Modern AI voice synthesis tools (ElevenLabs, XTTS-v2, Vall-E, Google TTS) can clone any human voice from just **3 seconds** of reference audio. This has fueled a rise in:

- 💰 Voice-based financial fraud (vishing attacks)
- 🗳️ Political misinformation using fake audio
- 🆔 Identity theft and voice authentication bypass
- ⚖️ Tampered audio evidence

VoiceGuard provides an automated, explainable, real-time defense against these threats.

---

## ✨ Key Features

- **5 Complementary Spectral Features** — Mel-Spectrogram, MFCC+Δ+ΔΔ, CQCC, IMFCC, and Phase Spectrum analyzed simultaneously
- **Cross-Attention Fusion** — 4-head attention mechanism that learns which feature to trust most for each input
- **Transformer Encoder** — Captures temporal patterns across the audio clip
- **Adaptive Noise Handling** — Automatic SNR estimation and spectral denoising for real-world robustness
- **Real-Time Live Detection** — WebRTC microphone capture with WebSocket streaming, <500ms latency
- **File Upload Mode** — Analyze MP3, WAV, FLAC, and OGG files
- **Explainable Results** — Per-feature confidence scores, not just a black-box verdict
- **18 Language Support** — Full UI translation including English, Tamil, Hindi, Telugu, Kannada, Malayalam, Bengali, Marathi, Urdu, French, German, Spanish, Chinese, Arabic, Japanese, Korean, Portuguese, and Russian

---

## 🏗️ System Architecture

```
                         ┌─────────────────────┐
                         │   Raw Audio Input    │
                         │  (Upload or Live Mic) │
                         └──────────┬──────────┘
                                    ▼
                  ┌─────────────────────────────────┐
                  │   Adaptive Preprocessing          │
                  │  16kHz resample → SNR estimate →  │
                  │  Denoise (if SNR < 25dB)          │
                  └──────────────┬───────────────────┘
                                 ▼
        ┌────────────────────────────────────────────────┐
        │            Multi-Feature Extraction              │
        │  Mel-Spec │ MFCC+Δ+ΔΔ │ CQCC │ IMFCC │ Phase*    │
        └──────┬───────┬───────┬───────┬───────┬──────────┘
               ▼        ▼       ▼       ▼       ▼
        ┌──────────────────────────────────────────────┐
        │     5× CNN Branches (Conv2d→BN→GELU→Pool)     │
        │            → 128-dim embeddings                │
        └───────────────────┬──────────────────────────┘
                             ▼
              ┌───────────────────────────────┐
              │   Cross-Attention Fusion        │
              │      (4 heads, learns trust)     │
              └───────────────┬───────────────┘
                              ▼
              ┌───────────────────────────────┐
              │   Transformer Encoder (2 layers) │
              └───────────────┬───────────────┘
                              ▼
              ┌───────────────────────────────┐
              │   Classifier → REAL / FAKE       │
              │      + confidence score          │
              └───────────────────────────────┘
```

*Phase Spectrum is used only when SNR ≥ 15dB

---

## 📊 Model Specifications

| Parameter | Value |
|---|---|
| Total Parameters | 4,922,636 |
| Model Size | 18.78 MB |
| Input Sample Rate | 16,000 Hz |
| Input Duration | 3 seconds |
| Inference Time (CPU) | < 500 ms |
| Inference Time (GPU) | 30–80 ms |
| Expected Accuracy | 85–95% (training dependent) |

---

## 🛠️ Tech Stack

**Backend:** Python 3.11 · PyTorch · FastAPI · Uvicorn · Librosa · NoiseReduce · WebSockets

**Frontend:** HTML5 · CSS3 · JavaScript · WebRTC · Canvas API

**Training:** AdamW Optimizer · CosineAnnealingLR · Cross-Entropy Loss with Label Smoothing

**Dataset:** FOR (Fake-or-Real) Dataset + custom TTS-generated samples (edge-tts, gTTS, pyttsx3, ElevenLabs)

---

## 📁 Project Structure

```
voiceguard/
├── backend/
│   ├── main.py                  # FastAPI server & API endpoints
│   ├── model.py                 # Deep learning model architecture
│   ├── feature_extractor.py     # 5-feature extraction pipeline
│   ├── train.py                 # Training script
│   ├── generate_fake_voices.py  # Multi-engine TTS data generator
│   ├── generate_elevenlabs.py   # ElevenLabs API voice generator
│   └── requirements.txt
├── frontend/
│   └── deepfake_audio_detector.html   # Full web application
├── data/
│   ├── real/                    # Real voice training samples
│   └── fake/                    # Fake voice training samples
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- pip
- (Optional) NVIDIA GPU with CUDA for faster training

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/voiceguard.git
cd voiceguard

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux

# Install dependencies
cd backend
pip install -r requirements.txt
```

### Running the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at **http://localhost:8000**

### Training the Model

```bash
python train.py --real_dir "../data/real" --fake_dir "../data/fake" --epochs 10 --batch_size 32
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the web application |
| `/predict` | POST | Analyze an uploaded audio file |
| `/predict/chunk` | POST | Analyze a live audio chunk |
| `/ws/live` | WebSocket | Real-time streaming detection |
| `/health` | GET | Server health check |
| `/model/info` | GET | Model architecture details |

---

## 🌍 Real-World Applications

- **Banking & Finance** — Voice-based transaction authorization
- **Call Centers** — Identity verification
- **Media & Journalism** — Audio evidence verification
- **Personal Security** — Protection against phone scams

---

## 🎯 Sustainable Development Goals

| SDG | Contribution |
|---|---|
| **SDG 9** — Industry, Innovation & Infrastructure | Promotes secure AI-powered authentication |
| **SDG 16** — Peace, Justice & Strong Institutions | Combats deepfake fraud and misinformation |
| **SDG 17** — Partnerships for the Goals | Open-source collaboration in AI safety |

---

## 🔮 Future Work

- [ ] Integration with ASVspoof 2024 and AUDETER datasets
- [ ] Continual learning to adapt to new AI voice generators
- [ ] Mobile application (Android/iOS)
- [ ] Multimodal audio-visual deepfake detection
- [ ] Lightweight model compression for edge deployment

---

## 👤 Author

**Reshenther**
Department of Information Technology
SRM Valliammai Engineering College
Mini Project — IT3641 | 2025–2026

---

## 📄 License

This project was developed for academic purposes as part of an undergraduate mini project.

---

## 🙏 Acknowledgments

- [FOR (Fake-or-Real) Dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset) by Reimao & Tzerpos
- [ASVspoof Challenge Series](https://www.asvspoof.org) for benchmark methodology
- Department of Information Technology, SRM Valliammai Engineering College
