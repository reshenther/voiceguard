"""
model.py
────────
Multi-feature fusion deep learning model for deepfake audio detection.

Architecture:
  - 5 parallel CNN branches (one per feature type)
  - Cross-attention fusion layer
  - Transformer encoder for temporal modeling
  - Binary classification head (Real / Fake)

Designed to work with:
  - Mel-Spectrogram  (80, T)
  - MFCC + deltas    (120, T)
  - CQCC             (40, T)
  - IMFCC            (40, T)
  - Phase Spectrum   (256, T)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


# ─────────────────────────────────────────────
# BUILDING BLOCKS
# ─────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → GELU → optional residual."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        self.residual = None
        if in_ch != out_ch or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.residual:
            x = self.residual(x)
        return out + x if x.shape == out.shape else out


class FeatureBranch(nn.Module):
    """
    Single-feature CNN branch.
    Takes a 2D spectrogram-like input and produces a 1D embedding.
    """

    def __init__(self, in_channels: int, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1
            ConvBlock(1, 32, kernel=3, stride=1, padding=1),
            ConvBlock(32, 32, kernel=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            # Block 2
            ConvBlock(32, 64, kernel=3, stride=1, padding=1),
            ConvBlock(64, 64, kernel=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.15),

            # Block 3
            ConvBlock(64, 128, kernel=3, stride=1, padding=1),
            ConvBlock(128, 128, kernel=3, stride=2, padding=1),
            nn.Dropout2d(0.2),
        )
        # Adaptive pool → fixed size regardless of input shape
        self.pool = nn.AdaptiveAvgPool2d((4, 8))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 8, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, freq, time) → add channel dim → (B, 1, freq, time)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.net(x)
        x = self.pool(x)
        return self.proj(x)


# ─────────────────────────────────────────────
# CROSS-ATTENTION FUSION
# ─────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """
    Fuse N feature embeddings using cross-attention.
    Each feature attends to all others to learn inter-feature relationships.
    """

    def __init__(self, embed_dim: int = 128, n_heads: int = 4, n_features: int = 5):
        super().__init__()
        self.n_features = n_features
        self.embed_dim = embed_dim

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, features: list) -> torch.Tensor:
        """
        features: list of (B, embed_dim) tensors
        Returns: (B, n_features, embed_dim)
        """
        # Stack: (B, n_features, embed_dim)
        x = torch.stack(features, dim=1)

        # Self-attention across features
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


# ─────────────────────────────────────────────
# MAIN MODEL
# ─────────────────────────────────────────────

class DeepfakeAudioDetector(nn.Module):
    """
    Full multi-feature deepfake audio detection model.

    Input:  Dict with keys mel, mfcc, cqcc, imfcc, phase
    Output: (logits, feature_scores) where logits shape = (B, 2)
    """

    def __init__(
        self,
        embed_dim: int = 128,
        n_transformer_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # ── Per-feature CNN branches ──
        self.branches = nn.ModuleDict({
            "mel":   FeatureBranch(1, embed_dim),
            "mfcc":  FeatureBranch(1, embed_dim),
            "cqcc":  FeatureBranch(1, embed_dim),
            "imfcc": FeatureBranch(1, embed_dim),
            "phase": FeatureBranch(1, embed_dim),
        })

        # ── Cross-attention fusion ──
        self.fusion = CrossAttentionFusion(
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_features=5,
        )

        # ── Transformer temporal encoder ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)

        # ── Per-feature classifier heads (for interpretability) ──
        self.feature_heads = nn.ModuleDict({
            name: nn.Linear(embed_dim, 2)
            for name in ["mel", "mfcc", "cqcc", "imfcc", "phase"]
        })

        # ── Final classifier ──
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 5, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(embed_dim, 2),
        )

    def forward(
        self,
        features: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            features: dict of tensors with shape (B, freq, time)

        Returns:
            logits:         (B, 2) — [real_logit, fake_logit]
            feature_scores: dict of per-feature fake probabilities
        """
        feature_order = ["mel", "mfcc", "cqcc", "imfcc", "phase"]

        # ── Step 1: Per-feature CNN encoding ──
        embeddings = {}
        for name in feature_order:
            embeddings[name] = self.branches[name](features[name])

        # ── Step 2: Per-feature interpretability scores ──
        feature_scores = {}
        for name in feature_order:
            logit = self.feature_heads[name](embeddings[name])
            prob = F.softmax(logit, dim=-1)
            feature_scores[name] = prob[:, 1]  # Fake probability

        # ── Step 3: Cross-attention fusion ──
        emb_list = [embeddings[n] for n in feature_order]
        fused = self.fusion(emb_list)  # (B, 5, embed_dim)

        # ── Step 4: Transformer over fused features ──
        transformed = self.transformer(fused)  # (B, 5, embed_dim)

        # ── Step 5: Classify ──
        flat = transformed.reshape(transformed.size(0), -1)  # (B, 5 * embed_dim)
        logits = self.classifier(flat)  # (B, 2)

        return logits, feature_scores

    def predict(
        self,
        features: Dict[str, np.ndarray],
        device: str = "cpu",
    ) -> Dict:
        """
        Inference wrapper: takes numpy features, returns prediction dict.

        Args:
            features: dict of numpy arrays from feature_extractor
            device:   'cpu' or 'cuda'

        Returns:
            {
                prediction:      'REAL' or 'FAKE',
                fake_probability: float 0-1,
                real_probability: float 0-1,
                confidence:       float 0-1 (max of both probs),
                feature_scores: {mel, mfcc, cqcc, imfcc, phase}
            }
        """
        self.eval()
        with torch.no_grad():
            # Convert numpy → tensors with batch dim
            tensor_features = {
                k: torch.FloatTensor(v).unsqueeze(0).to(device)
                for k, v in features.items()
            }

            logits, feat_scores = self.forward(tensor_features)
            probs = F.softmax(logits, dim=-1)[0]  # (2,)

            real_prob = float(probs[0])
            fake_prob = float(probs[1])
            is_fake   = fake_prob > 0.5
            confidence = max(real_prob, fake_prob)

            return {
                "prediction":      "FAKE" if is_fake else "REAL",
                "fake_probability": round(fake_prob, 4),
                "real_probability": round(real_prob, 4),
                "confidence":       round(confidence, 4),
                "feature_scores": {
                    k: round(float(v[0]), 4)
                    for k, v in feat_scores.items()
                },
            }


# ─────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────

def load_model(weights_path: str = None, device: str = "cpu") -> DeepfakeAudioDetector:
    """
    Load model. If weights_path is None, returns untrained model
    (useful for architecture testing).

    For production: train on ASVspoof2019 + WaveFake datasets and
    save with torch.save(model.state_dict(), 'voiceguard_model.pt')
    """
    model = DeepfakeAudioDetector(
        embed_dim=128,
        n_transformer_layers=2,
        n_heads=4,
        dropout=0.3,
    ).to(device)

    if weights_path:
        try:
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"[VoiceGuard] Model loaded from {weights_path}")
        except FileNotFoundError:
            print(f"[VoiceGuard] WARNING: No weights found at {weights_path}")
            print("[VoiceGuard] Running with random weights — train the model first!")
        except Exception as e:
            print(f"[VoiceGuard] Error loading weights: {e}")
    else:
        print("[VoiceGuard] No weights path provided — using random weights")

    model.eval()
    return model


def get_model_info(model: DeepfakeAudioDetector) -> Dict:
    """Return model parameter count and architecture info."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters":     total_params,
        "trainable_parameters": trainable,
        "model_size_mb":        round(total_params * 4 / 1024 / 1024, 2),
        "embed_dim":            model.embed_dim,
        "branches":             list(model.branches.keys()),
    }
