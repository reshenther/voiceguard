"""
generate_fake_voices.py
───────────────────────
Automatically generates fake AI voices using multiple
free TTS engines to improve live detection accuracy.

Engines used:
  1. pyttsx3     - Windows built-in TTS (offline)
  2. gTTS        - Google Text-to-Speech (online)
  3. edge-tts    - Microsoft Edge TTS (online, best quality)

Usage:
  python generate_fake_voices.py --output "../data/fake" --count 500
"""

import os
import sys
import random
import asyncio
import argparse
import time

# ─────────────────────────────────────────────
# TEXT SAMPLES (varied sentences for diversity)
# ─────────────────────────────────────────────
SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world.",
    "Voice authentication systems must be robust and reliable.",
    "Deep learning models can detect subtle audio patterns.",
    "The weather today is sunny with a chance of clouds.",
    "Technology has changed the way we communicate.",
    "Please verify your identity using voice recognition.",
    "The audio signal contains important frequency information.",
    "Machine learning algorithms improve with more training data.",
    "Natural language processing enables human computer interaction.",
    "Security systems must protect against deepfake attacks.",
    "The spectrogram reveals hidden patterns in speech.",
    "Neural networks learn from examples and feedback.",
    "Voice cloning technology poses serious privacy concerns.",
    "Authentication requires both accuracy and speed.",
    "The model analyzes five different spectral features.",
    "Real voices have natural variation and imperfections.",
    "Synthetic voices often lack natural prosody and rhythm.",
    "Feature extraction converts audio into numerical data.",
    "The classifier outputs a probability between zero and one.",
    "Good morning, how are you doing today?",
    "Please leave a message after the beep.",
    "Your call is very important to us.",
    "Welcome to the voice authentication system.",
    "Thank you for using our service today.",
    "The quick fox ran across the open field.",
    "Scientists discovered a new method for energy storage.",
    "The annual report showed significant growth this quarter.",
    "Students gathered in the hall for the ceremony.",
    "The library closes at nine o clock every evening.",
    "Innovation drives progress in every industry.",
    "The conference will begin promptly at eight.",
    "Please enter your password to continue.",
    "Your account has been successfully verified.",
    "The system detected an unusual login attempt.",
    "Data privacy is a fundamental human right.",
    "The research paper was published in a top journal.",
    "Advanced algorithms can process millions of records.",
    "The new software update includes security patches.",
    "Collaboration between teams leads to better outcomes.",
    "The experiment yielded unexpected but promising results.",
    "Engineers worked through the night to fix the issue.",
    "The startup raised funding from multiple investors.",
    "Cloud computing enables scalable and flexible solutions.",
    "The professor explained the concept with clear examples.",
    "Children learn best through play and exploration.",
    "The museum exhibit attracted thousands of visitors.",
    "Fresh produce is available at the local market.",
    "The flight was delayed due to bad weather.",
    "Reading books expands your knowledge and vocabulary.",
]


def install_package(package):
    """Install a Python package if not available."""
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])


# ─────────────────────────────────────────────
# ENGINE 1: pyttsx3 (Offline - Windows TTS)
# ─────────────────────────────────────────────

def generate_pyttsx3(text, output_path, rate=150, volume=0.9):
    """Generate speech using Windows built-in TTS (offline)."""
    try:
        import pyttsx3
    except ImportError:
        install_package("pyttsx3")
        import pyttsx3

    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')

        # Randomly pick a voice for variety
        if voices:
            engine.setProperty('voice', random.choice(voices).id)

        # Vary rate and volume for diversity
        engine.setProperty('rate', rate + random.randint(-20, 20))
        engine.setProperty('volume', volume)

        engine.save_to_file(text, output_path)
        engine.runAndWait()
        engine.stop()

        return os.path.exists(output_path)
    except Exception as e:
        print(f"  pyttsx3 error: {e}")
        return False


# ─────────────────────────────────────────────
# ENGINE 2: gTTS (Google TTS - Online)
# ─────────────────────────────────────────────

def generate_gtts(text, output_path, lang='en'):
    """Generate speech using Google Text-to-Speech."""
    try:
        from gtts import gTTS
    except ImportError:
        install_package("gtts")
        from gtts import gTTS

    try:
        # Vary language accent for diversity
        accents = ['com', 'co.uk', 'com.au', 'co.in']
        tld = random.choice(accents)

        tts = gTTS(text=text, lang=lang, tld=tld, slow=False)
        tts.save(output_path)
        return os.path.exists(output_path)
    except Exception as e:
        print(f"  gTTS error: {e}")
        return False


# ─────────────────────────────────────────────
# ENGINE 3: edge-tts (Microsoft - Best Quality)
# ─────────────────────────────────────────────

async def generate_edge_tts_async(text, output_path, voice=None):
    """Generate speech using Microsoft Edge TTS (high quality)."""
    try:
        import edge_tts
    except ImportError:
        install_package("edge-tts")
        import edge_tts

    # High quality voices for diversity
    voices = [
        "en-US-AriaNeural",
        "en-US-GuyNeural",
        "en-US-JennyNeural",
        "en-US-ChristopherNeural",
        "en-GB-SoniaNeural",
        "en-GB-RyanNeural",
        "en-AU-NatashaNeural",
        "en-AU-WilliamNeural",
        "en-IN-NeerjaNeural",
        "en-CA-ClaraNeural",
    ]

    selected_voice = voice or random.choice(voices)

    try:
        communicate = edge_tts.Communicate(text, selected_voice)
        await communicate.save(output_path)
        return os.path.exists(output_path)
    except Exception as e:
        print(f"  edge-tts error: {e}")
        return False


def generate_edge_tts(text, output_path, voice=None):
    """Wrapper to run async edge-tts."""
    return asyncio.run(generate_edge_tts_async(text, output_path, voice))


# ─────────────────────────────────────────────
# MAIN GENERATOR
# ─────────────────────────────────────────────

def generate_fake_voices(output_dir, count=500, engines=None):
    """
    Generate fake voices using multiple TTS engines.

    Args:
        output_dir: Where to save the generated WAV/MP3 files
        count:      Total number of fake voice files to generate
        engines:    List of engines to use ['pyttsx3', 'gtts', 'edge']
    """
    os.makedirs(output_dir, exist_ok=True)

    if engines is None:
        engines = ['edge', 'gtts', 'pyttsx3']

    # Distribute count across engines
    per_engine = count // len(engines)
    remainder  = count % len(engines)

    total_generated = 0
    file_index      = 0

    print(f"\n{'='*50}")
    print(f"  VoiceGuard — Fake Voice Generator")
    print(f"{'='*50}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Total files: {count}")
    print(f"  Engines:     {', '.join(engines)}")
    print(f"{'='*50}\n")

    for eng_idx, engine in enumerate(engines):
        engine_count = per_engine + (1 if eng_idx < remainder else 0)
        print(f"\n[{engine.upper()}] Generating {engine_count} files...")

        for i in range(engine_count):
            text        = random.choice(SENTENCES)
            file_index += 1
            filename    = f"fake_{engine}_{file_index:04d}.wav"
            output_path = os.path.join(output_dir, filename)

            # Skip if already exists
            if os.path.exists(output_path):
                print(f"  [{file_index}/{count}] Already exists — skipping")
                total_generated += 1
                continue

            success = False

            if engine == 'pyttsx3':
                success = generate_pyttsx3(text, output_path)

            elif engine == 'gtts':
                mp3_path = output_path.replace('.wav', '.mp3')
                success  = generate_gtts(text, mp3_path)
                if success:
                    # Rename to keep consistent naming
                    os.rename(mp3_path, mp3_path)
                    filename    = f"fake_{engine}_{file_index:04d}.mp3"
                    output_path = os.path.join(output_dir, filename)
                    os.rename(mp3_path, output_path)

            elif engine == 'edge':
                success = generate_edge_tts(text, output_path)

            if success:
                total_generated += 1
                print(f"  [{file_index}/{count}] ✓ {filename}")
            else:
                print(f"  [{file_index}/{count}] ✗ Failed — skipping")

            # Small delay to avoid rate limiting
            if engine in ['gtts', 'edge']:
                time.sleep(0.3)

    print(f"\n{'='*50}")
    print(f"  Done! Generated {total_generated}/{count} fake voice files")
    print(f"  Saved to: {output_dir}")
    print(f"{'='*50}\n")

    return total_generated


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate fake AI voices for VoiceGuard training"
    )
    parser.add_argument(
        "--output", default="../data/fake",
        help="Output directory for fake voice files"
    )
    parser.add_argument(
        "--count", type=int, default=300,
        help="Number of fake voice files to generate"
    )
    parser.add_argument(
        "--engines", nargs="+",
        default=["edge", "gtts", "pyttsx3"],
        choices=["edge", "gtts", "pyttsx3"],
        help="TTS engines to use"
    )
    args = parser.parse_args()

    generate_fake_voices(
        output_dir=args.output,
        count=args.count,
        engines=args.engines,
    )
