"""
generate_elevenlabs.py
──────────────────────
Generates fake AI voices using ElevenLabs API
for VoiceGuard deepfake detection training.

Free tier: 10,000 characters/month
Each sentence ~50 chars → ~200 samples free per month

Usage:
  python generate_elevenlabs.py --api_key YOUR_KEY --output "../data/fake" --count 100
"""

import os
import time
import random
import argparse
import requests

# ─────────────────────────────────────────────
# SENTENCES FOR GENERATION
# ─────────────────────────────────────────────
SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world.",
    "Voice authentication systems must be robust.",
    "Deep learning models can detect subtle patterns.",
    "The weather today is sunny with a chance of clouds.",
    "Technology has changed the way we communicate.",
    "Please verify your identity using voice recognition.",
    "The audio signal contains important frequency data.",
    "Machine learning algorithms improve with more data.",
    "Natural language processing enables human interaction.",
    "Security systems must protect against deepfake attacks.",
    "The spectrogram reveals hidden patterns in speech.",
    "Neural networks learn from examples and feedback.",
    "Voice cloning technology poses serious privacy concerns.",
    "Authentication requires both accuracy and speed.",
    "Good morning, how are you doing today?",
    "Please leave a message after the beep.",
    "Your call is very important to us.",
    "Welcome to the voice authentication system.",
    "Thank you for using our service today.",
    "Innovation drives progress in every industry.",
    "The conference will begin promptly at eight.",
    "Please enter your password to continue.",
    "Your account has been successfully verified.",
    "The system detected an unusual login attempt.",
    "Data privacy is a fundamental human right.",
    "Advanced algorithms can process millions of records.",
    "The new software update includes security patches.",
    "Cloud computing enables scalable flexible solutions.",
    "The professor explained the concept with clear examples.",
    "Fresh produce is available at the local market.",
    "Reading books expands your knowledge and vocabulary.",
    "The experiment yielded unexpected but promising results.",
    "Engineers worked through the night to fix the issue.",
    "The startup raised funding from multiple investors.",
    "Children learn best through play and exploration.",
    "The museum exhibit attracted thousands of visitors.",
    "The flight was delayed due to bad weather.",
    "Scientists discovered a new method for energy storage.",
    "The annual report showed significant growth this year.",
    "Students gathered in the hall for the ceremony.",
    "The library closes at nine o clock every evening.",
    "Collaboration between teams leads to better outcomes.",
    "The research paper was published in a top journal.",
    "Please speak clearly into the microphone now.",
    "Your voice sample has been recorded successfully.",
    "The detection system is analyzing your audio input.",
    "This is a test of the emergency broadcast system.",
    "Have a wonderful day and stay safe everyone.",
    "The model achieved high accuracy on the test set.",
]

# ─────────────────────────────────────────────
# ELEVENLABS VOICES (Free tier voices)
# ─────────────────────────────────────────────
VOICES = {
    "Rachel":      "21m00Tcm4TlvDq8ikWAM",
    "Drew":        "29vD33N1CtxCmqQRPOHJ",
    "Clyde":       "2EiwWnXFnvU5JabPnv8n",
    "Paul":        "5Q0t7uMcjvnagumLfvZi",
    "Domi":        "AZnzlk1XvdvUeBnXmlld",
    "Dave":        "CYw3kZ78EiZjFZnvdkwt",
    "Fin":         "D38z5RcWu1voky8WS1ja",
    "Bella":       "EXAVITQu4vr4xnSDxMaL",
    "Antoni":      "ErXwobaYiN019PkySvjV",
    "Thomas":      "GBv7mTt0atIp3Br8iCZE",
}


def generate_elevenlabs_voice(
    text: str,
    output_path: str,
    api_key: str,
    voice_id: str,
    model: str = "eleven_flash_v2_5",
    stability: float = 0.5,
    similarity_boost: float = 0.75,
) -> bool:
    """Generate a single voice sample using ElevenLabs API."""

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key,
    }

    payload = {
        "text": text,
        "model_id": model,
        "voice_settings": {
            "stability": stability + random.uniform(-0.1, 0.1),
            "similarity_boost": similarity_boost + random.uniform(-0.1, 0.1),
        }
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)

        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
            return os.path.exists(output_path) and os.path.getsize(output_path) > 0

        elif response.status_code == 401:
            print("  ERROR: Invalid API key! Check your ElevenLabs API key.")
            return False

        elif response.status_code == 429:
            print("  Rate limited — waiting 60 seconds...")
            time.sleep(60)
            return False

        elif response.status_code == 422:
            print(f"  Quota exceeded — free tier limit reached!")
            return False

        else:
            print(f"  API error: {response.status_code} — {response.text[:100]}")
            return False

    except requests.exceptions.Timeout:
        print("  Request timed out — skipping")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def check_quota(api_key: str) -> dict:
    """Check remaining ElevenLabs API quota."""
    try:
        response = requests.get(
            "https://api.elevenlabs.io/v1/user/subscription",
            headers={"xi-api-key": api_key},
            timeout=10,
        )
        if response.status_code == 200:
            data = response.json()
            used      = data.get("character_count", 0)
            limit     = data.get("character_limit", 10000)
            remaining = limit - used
            return {
                "used": used,
                "limit": limit,
                "remaining": remaining,
                "ok": remaining > 0,
            }
    except Exception:
        pass
    return {"used": 0, "limit": 10000, "remaining": 10000, "ok": True}


def generate_all(api_key: str, output_dir: str, count: int):
    """Generate fake voices using multiple ElevenLabs voices."""

    os.makedirs(output_dir, exist_ok=True)

    # Check quota first
    quota = check_quota(api_key)
    print(f"\n{'='*55}")
    print(f"  ElevenLabs Fake Voice Generator")
    print(f"{'='*55}")
    print(f"  API Quota:  {quota['used']:,} / {quota['limit']:,} chars used")
    print(f"  Remaining:  {quota['remaining']:,} characters")
    print(f"  Can generate ~{quota['remaining'] // 50} samples")
    print(f"  Output dir: {output_dir}")
    print(f"  Requested:  {count} samples")
    print(f"{'='*55}\n")

    if not quota["ok"]:
        print("ERROR: No quota remaining! Wait until next month.")
        return 0

    # Estimate if we have enough quota
    est_chars = count * 50
    if est_chars > quota["remaining"]:
        safe_count = quota["remaining"] // 50
        print(f"WARNING: Requested {count} but only have quota for ~{safe_count}")
        print(f"Generating {safe_count} samples instead...\n")
        count = safe_count

    voice_names  = list(VOICES.keys())
    total_generated = 0
    chars_used   = 0

    for i in range(count):
        # Rotate through voices for maximum diversity
        voice_name = voice_names[i % len(voice_names)]
        voice_id   = VOICES[voice_name]
        text       = random.choice(SENTENCES)

        # Vary stability for more diverse outputs
        stability        = random.uniform(0.3, 0.8)
        similarity_boost = random.uniform(0.5, 0.9)

        filename    = f"fake_elevenlabs_{voice_name.lower()}_{i+1:04d}.mp3"
        output_path = os.path.join(output_dir, filename)

        # Skip if already exists
        if os.path.exists(output_path):
            print(f"  [{i+1}/{count}] Already exists — skipping: {filename}")
            total_generated += 1
            continue

        print(f"  [{i+1}/{count}] Voice: {voice_name:10s} | Text: {text[:40]}...")

        success = generate_elevenlabs_voice(
            text=text,
            output_path=output_path,
            api_key=api_key,
            voice_id=voice_id,
            stability=stability,
            similarity_boost=similarity_boost,
        )

        if success:
            total_generated += 1
            chars_used += len(text)
            print(f"           ✓ Saved: {filename}")
        else:
            print(f"           ✗ Failed: {filename}")

        # Respect rate limits — 1 request per second
        time.sleep(1.2)

        # Stop if quota getting low
        if chars_used > quota["remaining"] - 100:
            print("\nWARNING: Approaching quota limit — stopping early!")
            break

    print(f"\n{'='*55}")
    print(f"  Done! Generated: {total_generated}/{count} files")
    print(f"  Characters used: ~{chars_used:,}")
    print(f"  Remaining quota: ~{quota['remaining'] - chars_used:,} chars")
    print(f"  Saved to: {output_dir}")
    print(f"{'='*55}\n")

    return total_generated


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ElevenLabs fake voices for VoiceGuard training"
    )
    parser.add_argument(
        "--api_key", required=True,
        help="Your ElevenLabs API key"
    )
    parser.add_argument(
        "--output", default="../data/fake",
        help="Output directory for fake voice files"
    )
    parser.add_argument(
        "--count", type=int, default=100,
        help="Number of fake voice files to generate"
    )
    args = parser.parse_args()

    generate_all(
        api_key=args.api_key,
        output_dir=args.output,
        count=args.count,
    )
