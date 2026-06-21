"""
train.py
────────
Training script for VoiceGuard deepfake audio detector.

Recommended datasets:
  - ASVspoof 2019 LA (primary)     → https://datashare.ed.ac.uk/handle/10283/3336
  - WaveFake                       → https://github.com/RUB-SysSec/WaveFake
  - In-the-Wild                    → https://deepfake-demo.aisec.fraunhofer.de

Usage:
  python train.py --real_dir data/real --fake_dir data/fake --epochs 50

Directory structure expected:
  data/
    real/   *.wav / *.mp3 / *.flac  (genuine human voices)
    fake/   *.wav / *.mp3 / *.flac  (AI-generated voices)
"""

import os
import argparse
import random
import logging
import hashlib
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import librosa

from feature_extractor import (
    load_audio, denoise, pad_or_trim,
    extract_mel_spectrogram, extract_mfcc,
    extract_cqcc, extract_imfcc, extract_phase_spectrum,
    estimate_snr, SAMPLE_RATE, DURATION, HOP_LENGTH,
    N_MELS, N_MFCC, N_CQCC, N_IMFCC,
)
from model import DeepfakeAudioDetector, get_model_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")

# Expected time dimension for all features after pad_or_trim (= 376)
EXPECTED_T = 1 + int(SAMPLE_RATE * DURATION) // HOP_LENGTH


def normalize_feature(arr: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Pad or trim a 2D feature array to exactly (rows, cols)."""
    arr = arr.astype(np.float32)
    # Time dimension (cols)
    if arr.shape[1] < cols:
        arr = np.pad(arr, ((0, 0), (0, cols - arr.shape[1])))
    else:
        arr = arr[:, :cols]
    # Frequency dimension (rows)
    if arr.shape[0] < rows:
        arr = np.pad(arr, ((0, rows - arr.shape[0]), (0, 0)))
    else:
        arr = arr[:rows]
    return arr


# ─────────────────────────────────────────────
# AUGMENTATION
# ─────────────────────────────────────────────

def augment_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """Random augmentation for training robustness."""
    aug = audio.copy()

    # Gaussian noise (simulate noisy environment)
    if random.random() < 0.4:
        noise_level = random.uniform(0.001, 0.025)
        aug = aug + noise_level * np.random.randn(len(aug)).astype(np.float32)

    # Time stretch (simulate speed variation)
    if random.random() < 0.3:
        rate = random.uniform(0.85, 1.15)
        aug = librosa.effects.time_stretch(aug, rate=rate)

    # Pitch shift (simulate different speakers)
    if random.random() < 0.3:
        steps = random.uniform(-2, 2)
        aug = librosa.effects.pitch_shift(aug, sr=sr, n_steps=steps)

    # Volume scaling
    if random.random() < 0.5:
        scale = random.uniform(0.6, 1.4)
        aug = aug * scale

    # Clip
    aug = np.clip(aug, -1.0, 1.0)
    return aug.astype(np.float32)


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}


def gather_files(directory: str, label: int) -> List[Tuple[str, int]]:
    files = []
    for p in Path(directory).rglob("*"):
        if p.suffix.lower() in AUDIO_EXTS:
            files.append((str(p), label))
    return files


def _cache_path(audio_path: str, cache_dir: str) -> str:
    """Generate a unique .npz cache filename for an audio file."""
    key = hashlib.md5(audio_path.encode()).hexdigest()
    return os.path.join(cache_dir, f"{key}.npz")


def precompute_features(samples: List[Tuple[str, int]], cache_dir: str) -> None:
    """
    Pre-extract features for all samples and save to .npz files.
    Run once before training — makes GPU training 5-10x faster.
    """
    os.makedirs(cache_dir, exist_ok=True)
    total   = len(samples)
    skipped = 0

    logger.info(f"Pre-computing features for {total} samples → {cache_dir}")
    logger.info("This runs ONCE. Subsequent training epochs load from cache (fast).")

    for i, (path, label) in enumerate(samples):
        out_path = _cache_path(path, cache_dir)
        if os.path.exists(out_path):
            skipped += 1
            if (i + 1) % 1000 == 0:
                logger.info(f"  [{i+1}/{total}] {skipped} already cached, skipping...")
            continue

        try:
            audio, sr = load_audio(path)
            snr_db    = estimate_snr(audio, sr)
            if snr_db < 25.0:
                audio = denoise(audio, sr)
            audio = pad_or_trim(audio, sr)

            mel   = extract_mel_spectrogram(audio, sr)
            mfcc  = extract_mfcc(audio, sr)
            cqcc  = extract_cqcc(audio, sr)
            imfcc = extract_imfcc(audio, sr)
            phase = extract_phase_spectrum(audio, sr) if snr_db >= 15.0 else \
                    np.zeros((256, mel.shape[1]), dtype=np.float32)

            np.savez_compressed(out_path,
                mel=mel, mfcc=mfcc, cqcc=cqcc, imfcc=imfcc,
                phase=phase, label=np.array(label), snr=np.array(snr_db))

        except Exception as e:
            logger.warning(f"  Failed to cache {path}: {e}")

        if (i + 1) % 500 == 0:
            logger.info(f"  [{i+1}/{total}] cached ({skipped} already existed)")

    logger.info(f"Feature caching complete! {total - skipped} new + {skipped} existing.")


class CachedDataset(Dataset):
    """
    Fast dataset that loads pre-computed .npz feature files.
    GPU stays fed — no CPU feature extraction during training.
    Label: 0 = Real, 1 = Fake
    """

    def __init__(self, samples: List[Tuple[str, int]], cache_dir: str, augment: bool = False):
        self.samples   = samples
        self.cache_dir = cache_dir
        self.augment   = augment
        logger.info(f"CachedDataset: {len(samples)} samples | augment={augment}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        cache_file  = _cache_path(path, self.cache_dir)

        try:
            data  = np.load(cache_file)
            mel   = normalize_feature(data['mel'],   N_MELS,      EXPECTED_T)
            mfcc  = normalize_feature(data['mfcc'],  N_MFCC * 3,  EXPECTED_T)
            cqcc  = normalize_feature(data['cqcc'],  N_CQCC,      EXPECTED_T)
            imfcc = normalize_feature(data['imfcc'], N_IMFCC,     EXPECTED_T)
            phase = normalize_feature(data['phase'], 256,         EXPECTED_T)

            # Light augmentation on cached features (no expensive librosa ops)
            if self.augment:
                noise = np.random.uniform(0.0, 0.02)
                mel   = mel  + noise * np.random.randn(*mel.shape).astype(np.float32)
                mfcc  = mfcc + noise * np.random.randn(*mfcc.shape).astype(np.float32)

        except Exception:
            # Fallback: correct-shaped zeros for corrupted/missing cache files
            logger.warning(f"Cache miss for {path} — using zeros")
            mel   = np.zeros((N_MELS,     EXPECTED_T), dtype=np.float32)
            mfcc  = np.zeros((N_MFCC * 3, EXPECTED_T), dtype=np.float32)
            cqcc  = np.zeros((N_CQCC,     EXPECTED_T), dtype=np.float32)
            imfcc = np.zeros((N_IMFCC,    EXPECTED_T), dtype=np.float32)
            phase = np.zeros((256,        EXPECTED_T), dtype=np.float32)

        return {
            "mel":   torch.FloatTensor(mel),
            "mfcc":  torch.FloatTensor(mfcc),
            "cqcc":  torch.FloatTensor(cqcc),
            "imfcc": torch.FloatTensor(imfcc),
            "phase": torch.FloatTensor(phase),
        }, torch.tensor(label, dtype=torch.long)


class AudioDataset(Dataset):
    """
    Fallback dataset — extracts features on-the-fly (slow, CPU-bound).
    Use CachedDataset for GPU training instead.
    """

    def __init__(self, samples: List[Tuple[str, int]], augment: bool = False):
        self.samples = samples
        self.augment = augment
        logger.info(f"AudioDataset (on-the-fly): {len(samples)} samples | augment={augment}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]

        try:
            audio, sr = load_audio(path)
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e} — using silence")
            audio = np.zeros(int(SAMPLE_RATE * 3), dtype=np.float32)
            sr = SAMPLE_RATE

        if self.augment:
            audio = augment_audio(audio, sr)

        snr_db = estimate_snr(audio, sr)
        if snr_db < 25.0:
            audio = denoise(audio, sr)
        audio = pad_or_trim(audio, sr)

        mel   = extract_mel_spectrogram(audio, sr)
        mfcc  = extract_mfcc(audio, sr)
        cqcc  = extract_cqcc(audio, sr)
        imfcc = extract_imfcc(audio, sr)
        phase = extract_phase_spectrum(audio, sr) if snr_db >= 15.0 else \
                np.zeros_like(extract_phase_spectrum(audio, sr))

        return {
            "mel":   torch.FloatTensor(mel),
            "mfcc":  torch.FloatTensor(mfcc),
            "cqcc":  torch.FloatTensor(cqcc),
            "imfcc": torch.FloatTensor(imfcc),
            "phase": torch.FloatTensor(phase),
        }, torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def compute_eer(real_scores: List[float], fake_scores: List[float]) -> float:
    """
    Compute Equal Error Rate (EER).
    Lower is better. Industry standard metric for spoof detection.
    """
    all_scores = real_scores + fake_scores
    all_labels = [0] * len(real_scores) + [1] * len(fake_scores)

    thresholds = sorted(set(all_scores))
    min_diff = float('inf')
    eer = 0.5

    for thr in thresholds:
        far = sum(1 for s, l in zip(all_scores, all_labels) if s >= thr and l == 0) / max(len(real_scores), 1)
        frr = sum(1 for s, l in zip(all_scores, all_labels) if s < thr and l == 1) / max(len(fake_scores), 1)
        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            eer = (far + frr) / 2

    return eer


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for batch_idx, (features, labels) in enumerate(loader):
        features = {k: v.to(device) for k, v in features.items()}
        labels   = labels.to(device)

        optimizer.zero_grad()
        logits, _ = model(features)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        if (batch_idx + 1) % 20 == 0:
            logger.info(
                f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                f"loss={total_loss/(batch_idx+1):.4f} "
                f"acc={correct/total:.1%}"
            )

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss  = 0.0
    correct     = 0
    total       = 0
    real_scores = []
    fake_scores = []

    # Free any leftover VRAM from training before validation
    if device == "cuda":
        torch.cuda.empty_cache()

    for features, labels in loader:
        features = {k: v.to(device) for k, v in features.items()}
        labels   = labels.to(device)

        logits, _ = model(features)
        loss      = criterion(logits, labels)
        probs     = torch.softmax(logits, dim=-1)

        total_loss += loss.item()
        preds   = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        # Move results to CPU immediately to free GPU memory
        for prob, lbl in zip(probs.cpu().numpy(), labels.cpu().numpy()):
            (fake_scores if lbl == 1 else real_scores).append(float(prob[1]))

        # Free GPU memory after each batch
        del features, labels, logits, loss, probs
        if device == "cuda":
            torch.cuda.empty_cache()

    eer = compute_eer(real_scores, fake_scores) if real_scores and fake_scores else 0.5
    return total_loss / len(loader), correct / total, eer


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train VoiceGuard deepfake detector")
    parser.add_argument("--real_dir",       required=True,        help="Path to real voice audio files")
    parser.add_argument("--fake_dir",       required=True,        help="Path to fake/AI voice audio files")
    parser.add_argument("--output",         default="voiceguard_model.pt", help="Output model path")
    parser.add_argument("--epochs",         type=int,   default=50)
    parser.add_argument("--batch_size",     type=int,   default=32)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--val_split",      type=float, default=0.15)
    parser.add_argument("--workers",        type=int,   default=0)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--max_per_class",  type=int,   default=30000,
                        help="Max samples per class. Use -1 for ALL data.")
    parser.add_argument("--cache_dir",      default="../data/feature_cache",
                        help="Directory to store pre-computed feature .npz files.")
    parser.add_argument("--no_cache",       action="store_true",
                        help="Disable feature caching (slow, not recommended).")
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on: {device}")

    # ── Gather files ──
    real_samples = gather_files(args.real_dir, label=0)
    fake_samples = gather_files(args.fake_dir, label=1)

    if not real_samples:
        raise ValueError(f"No audio files found in real_dir: {args.real_dir}")
    if not fake_samples:
        raise ValueError(f"No audio files found in fake_dir: {args.fake_dir}")

    logger.info(f"Real samples: {len(real_samples)} | Fake samples: {len(fake_samples)}")

    # Balance classes and apply per-class cap
    min_count = min(len(real_samples), len(fake_samples))
    cap       = min_count if args.max_per_class == -1 else min(min_count, args.max_per_class)
    logger.info(f"Using {cap} samples per class ({cap*2} total) — set --max_per_class -1 for all data")
    real_samples = random.sample(real_samples, cap)
    fake_samples = random.sample(fake_samples, cap)
    all_samples  = real_samples + fake_samples
    random.shuffle(all_samples)
    
    # Train/val split
    n_val    = int(len(all_samples) * args.val_split)
    val_data = all_samples[:n_val]
    trn_data = all_samples[n_val:]

    logger.info(f"Train: {len(trn_data)} | Val: {len(val_data)}")

    # ── Feature Caching (pre-compute once, train fast) ──
    use_cache = not args.no_cache
    if use_cache:
        logger.info(f"\n[CACHE] Pre-computing features → {args.cache_dir}")
        logger.info("[CACHE] First run takes time. All future runs load instantly from cache.")
        precompute_features(all_samples, args.cache_dir)
        trn_dataset = CachedDataset(trn_data, args.cache_dir, augment=True)
        val_dataset = CachedDataset(val_data, args.cache_dir, augment=False)
    else:
        logger.warning("[SLOW] Cache disabled — features extracted on-the-fly each batch!")
        trn_dataset = AudioDataset(trn_data, augment=True)
        val_dataset = AudioDataset(val_data, augment=False)

    trn_loader = DataLoader(
        trn_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device == "cuda"),
    )
    # Val loader uses smaller batch size and no pin_memory to avoid CUDA OOM
    val_batch = max(8, args.batch_size // 2)
    val_loader = DataLoader(
        val_dataset, batch_size=val_batch, shuffle=False,
        num_workers=args.workers, pin_memory=False,
    )

    # ── Model ──
    model = DeepfakeAudioDetector(embed_dim=128, n_transformer_layers=2).to(device)
    info  = get_model_info(model)
    logger.info(f"Model: {info['total_parameters']:,} params ({info['model_size_mb']}MB)")

    # ── Training setup ──
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_eer   = 1.0
    best_epoch = 0
    start_epoch = 1

    # ── Resume from checkpoint if available ──
    ckpt_path = args.output.replace(".pt", "_checkpoint.pt")
    if os.path.exists(ckpt_path):
        logger.info(f"⟳ Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_eer    = ckpt.get("best_eer", 1.0)
        best_epoch  = ckpt.get("best_epoch", 0)
        logger.info(f"  Resumed at epoch {start_epoch}/{args.epochs} | Best EER so far: {best_eer:.1%}")
    else:
        logger.info("Starting fresh training (no checkpoint found)")

    # ── Training loop ──
    for epoch in range(start_epoch, args.epochs + 1):
        logger.info(f"\n{'─'*50}")
        logger.info(f"EPOCH {epoch}/{args.epochs} | LR={scheduler.get_last_lr()[0]:.2e}")

        trn_loss, trn_acc = train_one_epoch(
            model, trn_loader, optimizer, criterion, device, epoch
        )
        val_loss, val_acc, val_eer = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            f"Train: loss={trn_loss:.4f} acc={trn_acc:.1%} | "
            f"Val: loss={val_loss:.4f} acc={val_acc:.1%} EER={val_eer:.1%}"
        )

        # Save best model
        if val_eer < best_eer:
            best_eer   = val_eer
            best_epoch = epoch
            torch.save(model.state_dict(), args.output)
            logger.info(f"✓ Best model saved → {args.output} (EER={best_eer:.1%})")

        # Save checkpoint after every epoch (enables resume if interrupted)
        torch.save({
            "epoch":      epoch,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict(),
            "best_eer":   best_eer,
            "best_epoch": best_epoch,
        }, ckpt_path)
        logger.info(f"  Checkpoint saved → {ckpt_path}")

    # Clean up checkpoint on successful completion
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        logger.info("Checkpoint removed (training complete)")

    logger.info(f"\n{'═'*50}")
    logger.info(f"Training complete!")
    logger.info(f"Best EER: {best_eer:.1%} at epoch {best_epoch}")
    logger.info(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
