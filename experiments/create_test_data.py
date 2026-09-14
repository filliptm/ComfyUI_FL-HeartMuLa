"""
Create synthetic test data for LoRA training.

This script generates a simple audio file with corresponding lyrics and tags
for testing the training pipeline.
"""

import torch
import numpy as np
from pathlib import Path
import wave
import struct

DATA_DIR = Path(__file__).parent / "data"


def create_test_audio():
    """Create a simple test audio file - a chord progression with harmonics."""

    sample_rate = 48000
    duration = 10  # 10 seconds

    t = torch.linspace(0, duration, int(sample_rate * duration))

    # Create a simple melody with multiple frequencies
    frequencies = [261.63, 329.63, 392.00]  # C4, E4, G4 (C major chord)

    audio = torch.zeros_like(t)
    for i, freq in enumerate(frequencies):
        # Add fundamental and harmonics
        audio += 0.3 * torch.sin(2 * np.pi * freq * t)
        audio += 0.15 * torch.sin(2 * np.pi * freq * 2 * t)  # 1st harmonic
        audio += 0.08 * torch.sin(2 * np.pi * freq * 3 * t)  # 2nd harmonic

    # Add some variation over time (simple envelope)
    envelope = torch.ones_like(t)
    envelope[:int(0.1 * sample_rate)] = torch.linspace(0, 1, int(0.1 * sample_rate))
    envelope[-int(0.5 * sample_rate):] = torch.linspace(1, 0, int(0.5 * sample_rate))
    audio = audio * envelope

    # Normalize
    audio = audio / audio.abs().max() * 0.8

    # Add slight noise for texture
    audio += torch.randn_like(audio) * 0.01

    # Ensure shape is [channels, samples]
    audio = audio.unsqueeze(0)

    return audio, sample_rate


def create_test_lyrics():
    """Create simple test lyrics with section markers."""

    lyrics = """[Verse]
la la la
singing a song
la la la
all day long

[Chorus]
oh oh oh
here we go
oh oh oh
let it flow"""

    return lyrics


def create_test_tags():
    """Create test style tags."""

    tags = "pop, female vocal, energetic, happy, medium tempo"

    return tags


def save_wav(filepath, audio_tensor, sample_rate):
    """Save audio tensor as WAV file using standard library."""
    # Convert to numpy and ensure correct shape
    if isinstance(audio_tensor, torch.Tensor):
        audio = audio_tensor.numpy()
    else:
        audio = audio_tensor

    # Ensure 1D or get first channel
    if audio.ndim == 2:
        audio = audio[0]

    # Normalize to int16 range
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)

    # Write WAV file
    with wave.open(str(filepath), 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 2 bytes = 16 bits
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())


def main():
    """Generate all test data."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Creating synthetic test data...")

    # Create audio
    audio, sr = create_test_audio()
    audio_path = DATA_DIR / "test_sample.wav"
    save_wav(audio_path, audio, sr)
    print(f"Created: {audio_path}")
    print(f"  Duration: {audio.shape[1] / sr:.2f} seconds")
    print(f"  Sample rate: {sr} Hz")

    # Create lyrics
    lyrics = create_test_lyrics()
    lyrics_path = DATA_DIR / "test_sample.txt"
    lyrics_path.write_text(lyrics, encoding='utf-8')
    print(f"Created: {lyrics_path}")

    # Create tags
    tags = create_test_tags()
    tags_path = DATA_DIR / "test_sample.tags"
    tags_path.write_text(tags, encoding='utf-8')
    print(f"Created: {tags_path}")

    print("\nTest data created successfully!")
    print(f"Directory: {DATA_DIR}")


if __name__ == "__main__":
    main()
