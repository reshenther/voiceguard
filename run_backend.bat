@echo off

cd /d "C:\Users\Reshenther\voice authentix"

call venv312\Scripts\activate

"C:\Users\Reshenther\voice authentix\venv312\Scripts\python.exe" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload