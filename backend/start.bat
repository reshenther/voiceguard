@echo off
title VoiceGuard Backend Server
color 0B

echo.
echo  ============================================
echo    VoiceGuard — Deepfake Audio Detector
echo    Backend Server Starting...
echo  ============================================
echo.

cd /d "C:\Users\Reshenther\voice authentix"

call "C:\Users\Reshenther\voice authentix\venv312\Scripts\activate.bat"

echo  [OK] Virtual environment activated (venv312 - Python 3.12)
echo  [OK] Starting FastAPI server on port 8000...
echo.
echo  Open browser at: http://localhost:8000
echo  API docs at:     http://localhost:8000/docs
echo.
echo  Press CTRL+C to stop the server
echo  ============================================
echo.

"C:\Users\Reshenther\voice authentix\venv312\Scripts\python.exe" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

pause
