"""
main.py
───────
FastAPI backend for VoiceGuard Deepfake Audio Detector.

Endpoints:
  POST /predict          — Upload audio file for analysis
  POST /predict/chunk    — Submit raw audio bytes (live detection chunk)
  WS   /ws/live          — WebSocket for real-time streaming detection
  GET  /health           — Health check
  GET  /model/info       — Model architecture info
"""

import os
import io
import uuid
import time
import tempfile
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .feature_extractor import (
    preprocess_audio_file,
    preprocess_audio_chunk,
    estimate_snr,
    load_audio_bytes,
    SAMPLE_RATE,
)
from .model import load_model, get_model_info, DeepfakeAudioDetector

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("voiceguard")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_WEIGHTS   = os.getenv(
    "MODEL_WEIGHTS_PATH",
    os.path.join(os.path.dirname(__file__), "voiceguard_model.pt"),
)
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
MAX_FILE_SIZE   = 50 * 1024 * 1024   # 50 MB
ALLOWED_TYPES   = {
    "audio/mpeg", "audio/wav", "audio/wave", "audio/x-wav",
    "audio/ogg", "audio/flac", "audio/mp4", "audio/x-m4a",
    "audio/aac", "audio/webm", "application/octet-stream",
}
ALLOWED_EXTS    = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".webm"}


# ─────────────────────────────────────────────
# GLOBAL MODEL STATE
# ─────────────────────────────────────────────
class AppState:
    model: Optional[DeepfakeAudioDetector] = None
    device: str = DEVICE
    model_info: dict = {}
    startup_time: float = 0.0

app_state = AppState()


# ─────────────────────────────────────────────
# LIFESPAN (replaces @app.on_event)
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, cleanup at shutdown."""
    logger.info(f"Starting VoiceGuard on device: {DEVICE}")
    app_state.startup_time = time.time()
    app_state.model = load_model(weights_path=MODEL_WEIGHTS, device=DEVICE)
    app_state.model_info = get_model_info(app_state.model)
    logger.info(f"Model ready — {app_state.model_info['total_parameters']:,} parameters")
    yield
    logger.info("Shutting down VoiceGuard")


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────
app = FastAPI(
    title="VoiceGuard API",
    description="Deepfake audio detection using multi-feature spectrogram analysis",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────
class PredictionResponse(BaseModel):
    request_id:       str
    prediction:       str           # "REAL" or "FAKE"
    fake_probability: float
    real_probability: float
    confidence:       float
    snr_db:           float
    processing_ms:    float
    feature_scores:   dict          # {mel, mfcc, cqcc, imfcc, phase}
    noise_warning:    Optional[str] = None


class HealthResponse(BaseModel):
    status:       str
    device:       str
    uptime_s:     float
    model_loaded: bool


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def validate_file(file: UploadFile, content: bytes) -> None:
    """Validate file size and type."""
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, f"File too large. Max {MAX_FILE_SIZE // 1024 // 1024}MB.")

    ext = os.path.splitext(file.filename or "")[1].lower()
    ct  = (file.content_type or "").lower()

    if ext not in ALLOWED_EXTS and ct not in ALLOWED_TYPES:
        raise HTTPException(
            415,
            f"Unsupported format '{ext}'. Allowed: {', '.join(ALLOWED_EXTS)}",
        )


def run_inference(features: dict, snr_db: float) -> dict:
    """Run model inference and attach noise warning if needed."""
    result = app_state.model.predict(features, device=app_state.device)
    result["snr_db"] = round(snr_db, 2)

    if snr_db < 10:
        result["noise_warning"] = (
            f"High noise detected (SNR={snr_db:.1f}dB). "
            "Accuracy may be reduced. Phase spectrum feature disabled."
        )
    elif snr_db < 15:
        result["noise_warning"] = (
            f"Moderate noise (SNR={snr_db:.1f}dB). "
            "Phase spectrum feature disabled for reliability."
        )
    return result


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    return FileResponse("deepfake_audio_detector.html")
async def health_check():
    """Health check endpoint."""
    return {
        "status":       "ok",
        "device":       DEVICE,
        "uptime_s":     round(time.time() - app_state.startup_time, 1),
        "model_loaded": app_state.model is not None,
    }


@app.get("/model/info")
async def model_info():
    """Return model architecture details."""
    return {
        **app_state.model_info,
        "device": DEVICE,
        "features": ["Mel-Spectrogram", "MFCC+Δ+ΔΔ", "CQCC", "IMFCC", "Phase Spectrum"],
        "sample_rate": SAMPLE_RATE,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_file(file: UploadFile = File(...)):
    """
    Analyze an uploaded audio file for deepfake detection.

    Accepts: MP3, WAV, OGG, FLAC, M4A, AAC
    Returns: prediction, confidence, per-feature scores, SNR estimate
    """
    if app_state.model is None:
        raise HTTPException(503, "Model not loaded. Try again in a moment.")

    request_id = str(uuid.uuid4())[:8]
    t_start    = time.time()

    logger.info(f"[{request_id}] Received: {file.filename} ({file.content_type})")

    # Read and validate
    content = await file.read()
    validate_file(file, content)

    # Save to temp file for librosa
    ext = os.path.splitext(file.filename or "audio.wav")[1].lower() or ".wav"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Feature extraction
        features, snr_db = preprocess_audio_file(tmp_path)

        # Inference
        result = run_inference(features, snr_db)

        elapsed_ms = (time.time() - t_start) * 1000
        logger.info(
            f"[{request_id}] → {result['prediction']} "
            f"(conf={result['confidence']:.1%}, SNR={snr_db:.1f}dB, {elapsed_ms:.0f}ms)"
        )

        return PredictionResponse(
            request_id       = request_id,
            prediction       = result["prediction"],
            fake_probability = result["fake_probability"],
            real_probability = result["real_probability"],
            confidence       = result["confidence"],
            snr_db           = result["snr_db"],
            processing_ms    = round(elapsed_ms, 1),
            feature_scores   = result["feature_scores"],
            noise_warning    = result.get("noise_warning"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Processing failed: {e}", exc_info=True)
        raise HTTPException(500, f"Audio processing failed: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post("/predict/chunk")
async def predict_chunk(file: UploadFile = File(...)):
    """
    Analyze a raw audio chunk (for live detection mode).
    Optimized for low-latency processing of 2-3 second chunks.
    """
    if app_state.model is None:
        raise HTTPException(503, "Model not loaded.")

    t_start = time.time()
    content = await file.read()

    if len(content) < 1024:
        raise HTTPException(400, "Chunk too short (< 1KB). Wait for more audio.")

    try:
        features, snr_db = preprocess_audio_chunk(content)
        result           = run_inference(features, snr_db)
        elapsed_ms       = (time.time() - t_start) * 1000

        return {
            "prediction":       result["prediction"],
            "fake_probability": result["fake_probability"],
            "real_probability": result["real_probability"],
            "confidence":       result["confidence"],
            "snr_db":           round(snr_db, 1),
            "processing_ms":    round(elapsed_ms, 1),
            "feature_scores":   result["feature_scores"],
            "noise_warning":    result.get("noise_warning"),
        }

    except Exception as e:
        import traceback
        logger.warning(f"Chunk processing failed: {e}")
        logger.warning(traceback.format_exc())
        raise HTTPException(500, f"Chunk processing failed: {str(e)}")


# ─────────────────────────────────────────────
# WEBSOCKET — LIVE STREAMING
# ─────────────────────────────────────────────

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """
    WebSocket endpoint for real-time live audio detection.

    Protocol:
      Client → Server: raw audio bytes (binary frames)
      Server → Client: JSON prediction result

    The client should send audio chunks every 2-3 seconds.
    """
    await websocket.accept()
    client_id = str(uuid.uuid4())[:6]
    logger.info(f"[WS:{client_id}] Connected")

    buffer        = bytearray()
    chunk_count   = 0
    CHUNK_BYTES   = 48000 * 2 * 2  # ~3s @ 16kHz, 16-bit, mono (approx)

    try:
        while True:
            # Receive audio data
            try:
                data = await asyncio.wait_for(
                    websocket.receive_bytes(),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
                continue

            buffer.extend(data)

            # Process when we have enough data
            if len(buffer) >= CHUNK_BYTES:
                chunk_count += 1
                chunk_data   = bytes(buffer[:CHUNK_BYTES])
                buffer       = buffer[CHUNK_BYTES:]

                try:
                    t0 = time.time()
                    features, snr_db = preprocess_audio_chunk(chunk_data)
                    result           = run_inference(features, snr_db)
                    elapsed_ms       = (time.time() - t0) * 1000

                    await websocket.send_json({
                        "type":             "result",
                        "chunk":            chunk_count,
                        "prediction":       result["prediction"],
                        "fake_probability": result["fake_probability"],
                        "real_probability": result["real_probability"],
                        "confidence":       result["confidence"],
                        "snr_db":           round(snr_db, 1),
                        "processing_ms":    round(elapsed_ms, 1),
                        "feature_scores":   result["feature_scores"],
                        "noise_warning":    result.get("noise_warning"),
                    })

                    logger.info(
                        f"[WS:{client_id}] Chunk {chunk_count}: "
                        f"{result['prediction']} ({result['confidence']:.1%})"
                    )

                except Exception as e:
                    logger.warning(f"[WS:{client_id}] Chunk {chunk_count} failed: {e}")
                    await websocket.send_json({
                        "type":  "error",
                        "chunk": chunk_count,
                        "msg":   "Processing failed for this chunk",
                    })

    except WebSocketDisconnect:
        logger.info(f"[WS:{client_id}] Disconnected after {chunk_count} chunks")
    except Exception as e:
        logger.error(f"[WS:{client_id}] Error: {e}", exc_info=True)
        try:
            await websocket.close(code=1011, reason=str(e))
        except Exception:
            pass


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
