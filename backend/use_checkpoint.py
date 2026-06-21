"""
use_checkpoint.py
─────────────────
Extracts the model weights from voiceguard_model_checkpoint.pt
and saves them as voiceguard_model.pt so the backend can use them.
"""
import os
import torch

BACKEND_DIR   = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT    = os.path.join(BACKEND_DIR, "voiceguard_model_checkpoint.pt")
BEST_MODEL    = os.path.join(BACKEND_DIR, "voiceguard_model.pt")

print("=" * 50)

# Check what files exist
ckpt_exists  = os.path.exists(CHECKPOINT)
model_exists = os.path.exists(BEST_MODEL)

print(f"voiceguard_model.pt            exists: {model_exists}")
print(f"voiceguard_model_checkpoint.pt exists: {ckpt_exists}")

if model_exists:
    size_mb = os.path.getsize(BEST_MODEL) / (1024 * 1024)
    print(f"\nvoiceguard_model.pt is ready ({size_mb:.1f} MB)")
    print("The backend will load this model automatically.")

if ckpt_exists:
    ckpt    = torch.load(CHECKPOINT, map_location="cpu")
    epoch   = ckpt.get("epoch", "?")
    best_eer = ckpt.get("best_eer", None)
    print(f"\nCheckpoint is from epoch {epoch}")
    if best_eer:
        print(f"Best EER recorded: {best_eer:.1%}")

    # Extract and save model weights from checkpoint
    out_path = os.path.join(BACKEND_DIR, "voiceguard_model.pt")
    torch.save(ckpt["model"], out_path)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\n✅ Model weights extracted from checkpoint and saved to:")
    print(f"   {out_path}  ({size_mb:.1f} MB)")
    print("\nYou can now start the backend — it will load this model.")

if not ckpt_exists and not model_exists:
    print("\n❌ No model files found in backend directory.")
    print("   Training must be run first.")

print("=" * 50)
