"""
feature_extractor.py
────────────────────
Multi-feature extraction pipeline for deepfake audio detection.
Extracts: Mel-Spectrogram, MFCC, CQCC, IMFCC, Phase Spectrum

All features are normalized and returned as numpy arrays
ready for model inference.
"""

import numpy as np
import librosa
import noisereduce as nr
from scipy.fft import dct
from typing import Dict, Tuple
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SAMPLE_RATE     = 16000   # Standardize everything to 16kHz
DURATION        = 3.0     # Max seconds per chunk
N_FFT           = 512
HOP_LENGTH      = 128
N_MELS          = 80
N_MFCC          = 40
N_CQCC          = 40
N_IMFCC         = 40
FMIN            = 50.0
FMAX            = 8000.0


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────

def load_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """Load audio file and resample to 16kHz mono."""
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    return audio, sr


def load_audio_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Load audio from raw bytes (for live chunks)."""
    import io
    import tempfile
    import os

    # Try multiple formats — browser sends webm/opus
    errors = []

    # Method 1: soundfile
    try:
        import soundfile as sf
        audio, sr = sf.read(io.BytesIO(audio_bytes))
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        return audio.astype(np.float32), SAMPLE_RATE
    except Exception as e:
        errors.append(f"soundfile: {e}")

    # Method 2: librosa with temp webm file
    try:
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        try:
            audio, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
            return audio.astype(np.float32), SAMPLE_RATE
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        errors.append(f"librosa webm: {e}")

    # Method 3: librosa with temp wav file
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        try:
            audio, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
            return audio.astype(np.float32), SAMPLE_RATE
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        errors.append(f"librosa wav: {e}")

    # Method 4: return silence as fallback so app doesn't crash
    import logging
    logging.getLogger("voiceguard").warning(f"All audio load methods failed: {errors} — using silence")
    silence = np.zeros(int(SAMPLE_RATE * 3), dtype=np.float32)
    return silence, SAMPLE_RATE


def denoise(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Apply spectral noise reduction.
    Uses first 0.3s as noise profile estimate if audio is long enough,
    otherwise uses the whole clip for noise estimation.
    """
    try:
        noise_clip_len = int(0.3 * sr)
        if len(audio) > noise_clip_len * 2:
            noise_profile = audio[:noise_clip_len]
        else:
            noise_profile = audio

        denoised = nr.reduce_noise(
            y=audio,
            y_noise=noise_profile,
            sr=sr,
            prop_decrease=0.75,
            stationary=False,
        )
        return denoised.astype(np.float32)
    except Exception:
        return audio  # Return original if denoising fails


def pad_or_trim(audio: np.ndarray, sr: int, duration: float = DURATION) -> np.ndarray:
    """Pad or trim audio to fixed length."""
    target_len = int(sr * duration)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
    else:
        audio = audio[:target_len]
    return audio


def estimate_snr(audio: np.ndarray, sr: int) -> float:
    """
    Estimate Signal-to-Noise Ratio in dB.
    Uses RMS of signal vs estimated noise floor.
    """
    frame_len = int(0.025 * sr)  # 25ms frames
    hop = int(0.010 * sr)        # 10ms hop

    frames = librosa.util.frame(audio, frame_length=frame_len, hop_length=hop)
    rms_per_frame = np.sqrt(np.mean(frames ** 2, axis=0))

    # Noise floor = bottom 10% of frames
    noise_floor = np.percentile(rms_per_frame, 10)
    signal_rms  = np.percentile(rms_per_frame, 90)

    if noise_floor < 1e-8:
        return 40.0  # Very clean

    snr_db = 20 * np.log10((signal_rms + 1e-8) / (noise_floor + 1e-8))
    return float(np.clip(snr_db, 0, 50))


# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────

def extract_mel_spectrogram(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Mel-Spectrogram: Perceptually weighted frequency representation.
    Captures overall tonal structure of speech.
    Shape: (N_MELS, T)
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Normalize to [0, 1]
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_norm.astype(np.float32)


def extract_mfcc(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    MFCC: Mel Frequency Cepstral Coefficients.
    Captures vocal tract shape — AI voices have unnatural formant patterns.
    Shape: (N_MFCC, T)
    """
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        fmin=FMIN,
        fmax=FMAX,
    )
    # Add delta and delta-delta for temporal dynamics
    mfcc_delta  = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # Stack: shape (N_MFCC * 3, T)
    mfcc_full = np.vstack([mfcc, mfcc_delta, mfcc_delta2])

    # Normalize
    mean = mfcc_full.mean(axis=1, keepdims=True)
    std  = mfcc_full.std(axis=1, keepdims=True) + 1e-8
    return ((mfcc_full - mean) / std).astype(np.float32)


def extract_cqcc(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    CQCC: Constant-Q Cepstral Coefficients.
    Logarithmically spaced — more noise-robust than MFCC.
    Best for catching neural vocoder artifacts (WaveNet, HiFi-GAN).
    Shape: (N_CQCC, T)
    """
    # Safely calculate max bins that won't exceed Nyquist
    fmin = 60.0  # Hz — safe starting frequency
    nyquist = sr / 2.0
    bins_per_octave = 12
    n_octaves = int(np.floor(np.log2(nyquist / fmin))) - 1
    n_bins = min(n_octaves * bins_per_octave, 60)  # cap at 60 bins

    try:
        cqt = np.abs(librosa.cqt(
            y=audio,
            sr=sr,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            hop_length=HOP_LENGTH,
        ))
    except Exception:
        # Fallback: use STFT-based approximation
        stft = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH))
        cqt = stft[:n_bins, :]

    # Log compression
    log_cqt = np.log(cqt + 1e-8)

    # Cepstral transform via DCT
    cqcc = dct(log_cqt, axis=0, norm='ortho')[:N_CQCC]

    # Normalize
    mean = cqcc.mean(axis=1, keepdims=True)
    std  = cqcc.std(axis=1, keepdims=True) + 1e-8
    return ((cqcc - mean) / std).astype(np.float32)


def extract_imfcc(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    IMFCC: Inverted Mel Frequency Cepstral Coefficients.
    Focuses on HIGH-frequency bands (complement to MFCC).
    Catches synthesis artifacts that MFCC misses.
    Shape: (N_IMFCC, T)
    """
    # Inverted mel filterbank: high-frequency emphasis
    mel_inverted = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_IMFCC,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    # Flip frequency axis to invert emphasis
    mel_inv_db = librosa.power_to_db(mel_inverted[::-1], ref=np.max)

    # Apply DCT for cepstral coefficients
    imfcc = dct(mel_inv_db, axis=0, norm='ortho')[:N_IMFCC]

    # Normalize
    mean = imfcc.mean(axis=1, keepdims=True)
    std  = imfcc.std(axis=1, keepdims=True) + 1e-8
    return ((imfcc - mean) / std).astype(np.float32)


def extract_phase_spectrum(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Phase Spectrum + Group Delay.
    Real voices have structured phase patterns.
    AI vocoders destroy natural phase coherence.
    NOTE: Only reliable on clean/denoised audio (SNR > 15dB).
    Shape: (2 * n_bins, T)
    """
    stft = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    n_bins = N_FFT // 2 + 1

    # Instantaneous phase
    phase = np.angle(stft)

    # Unwrapped phase (remove 2π discontinuities)
    phase_unwrapped = np.unwrap(phase, axis=1)

    # Group delay (first derivative of phase w.r.t. frequency)
    group_delay = np.diff(phase_unwrapped, axis=1)
    group_delay = np.pad(group_delay, ((0, 0), (0, 1)), mode='edge')

    # Stack phase + group delay
    phase_features = np.vstack([
        phase_unwrapped[:n_bins // 2],  # Lower half to reduce size
        group_delay[:n_bins // 2],
    ])

    # Normalize
    mean = phase_features.mean(axis=1, keepdims=True)
    std  = phase_features.std(axis=1, keepdims=True) + 1e-8
    return ((phase_features - mean) / std).astype(np.float32)


# ─────────────────────────────────────────────
# UNIFIED PIPELINE
# ─────────────────────────────────────────────

def extract_all_features(
    audio: np.ndarray,
    sr: int,
    snr_db: float = 20.0,
    apply_denoise: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Full multi-feature extraction pipeline.

    Args:
        audio:          Raw audio waveform
        sr:             Sample rate (should be 16000)
        snr_db:         Estimated SNR in dB (disables phase if noisy)
        apply_denoise:  Whether to apply noise reduction first

    Returns:
        Dict with keys: mel, mfcc, cqcc, imfcc, phase
    """
    # Step 1: Pad/trim to fixed duration
    audio = pad_or_trim(audio, sr)

    # Step 2: Denoise if needed
    if apply_denoise and snr_db < 25.0:
        audio = denoise(audio, sr)

    # Step 3: Extract all features
    features = {
        "mel":   extract_mel_spectrogram(audio, sr),
        "mfcc":  extract_mfcc(audio, sr),
        "cqcc":  extract_cqcc(audio, sr),
        "imfcc": extract_imfcc(audio, sr),
    }

    # Phase spectrum only reliable above 15dB SNR
    if snr_db >= 15.0:
        features["phase"] = extract_phase_spectrum(audio, sr)
    else:
        # Use zeros placeholder — model trained to handle missing phase
        T = features["mel"].shape[1]
        features["phase"] = np.zeros((N_FFT // 2, T), dtype=np.float32)

    return features


def preprocess_audio_file(file_path: str) -> Tuple[Dict[str, np.ndarray], float]:
    """
    Complete preprocessing for an uploaded audio file.

    Returns:
        (features_dict, snr_db)
    """
    audio, sr = load_audio(file_path)
    snr_db = estimate_snr(audio, sr)
    features = extract_all_features(audio, sr, snr_db=snr_db, apply_denoise=True)
    return features, snr_db


def preprocess_audio_chunk(audio_bytes: bytes) -> Tuple[Dict[str, np.ndarray], float]:
    """
    Complete preprocessing for a live audio chunk (bytes).

    Returns:
        (features_dict, snr_db)
    """
    audio, sr = load_audio_bytes(audio_bytes)
    snr_db = estimate_snr(audio, sr)
    features = extract_all_features(audio, sr, snr_db=snr_db, apply_denoise=True)
    return features, snr_db
