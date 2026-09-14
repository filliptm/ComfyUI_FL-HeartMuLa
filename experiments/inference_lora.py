"""
HeartMuLa LoRA Inference Script

This script loads a trained LoRA checkpoint and generates audio.

Usage:
    Activate ComfyUI venv first:
    > d:\ComfyUI\venv\Scripts\activate.bat
    > python inference_lora.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).parent.absolute()
NODE_PACK_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(NODE_PACK_DIR))

import torch
import torch.nn as nn
import numpy as np
import wave
import scipy.io.wavfile as wavfile

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to LoRA checkpoint
LORA_CHECKPOINT = SCRIPT_DIR / "checkpoints" / "lora_final.pt"

# Generation parameters
OUTPUT_PATH = SCRIPT_DIR / "output_with_lora.wav"
MAX_DURATION_SEC = 30
TEMPERATURE = 1.0
TOP_K = 50
CFG_SCALE = 1.5

# Test prompt
TEST_TAGS = "Electronic"
TEST_LYRICS = """[Verse]
hello world
this is a test
of the lora
doing its best

[Chorus]
la la la
we trained a model
la la la
now it can sing"""

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

# ============================================================================
# IMPORTS
# ============================================================================

from heartlib.pipelines.music_generation import HeartMuLaGenPipeline
from heartlib.heartmula.modeling_heartmula import HeartMuLa
from fl_utils.model_manager import get_models_directory, download_models_if_needed, check_models_exist

# Import LoRA class from training script
from train_lora import LoRALinear, apply_lora_to_model, load_lora_state_dict


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("HeartMuLa LoRA Inference")
    print("=" * 60)

    # Check checkpoint exists
    if not LORA_CHECKPOINT.exists():
        print(f"ERROR: LoRA checkpoint not found at {LORA_CHECKPOINT}")
        print("Please run train_lora.py first to create a checkpoint.")
        return

    # Load checkpoint to get config
    print(f"\nLoading LoRA checkpoint: {LORA_CHECKPOINT}")
    checkpoint = torch.load(LORA_CHECKPOINT, map_location='cpu')
    lora_config = checkpoint['config']
    print(f"  Step: {checkpoint['step']}")
    print(f"  Rank: {lora_config['rank']}")
    print(f"  Alpha: {lora_config['alpha']}")
    print(f"  Parameters: {len(checkpoint['lora_state_dict'])}")

    # Load base model
    print("\nLoading HeartMuLa model...")

    if not check_models_exist("3B"):
        print("Model not found, downloading...")
        download_models_if_needed("3B")

    models_dir = get_models_directory()

    pipeline = HeartMuLaGenPipeline.from_pretrained(
        str(models_dir),
        device=torch.device(DEVICE),
        dtype=DTYPE,
        version="3B",
    )

    model = pipeline.model
    print(f"Model loaded on {DEVICE}")

    # Apply LoRA architecture
    print("\nApplying LoRA architecture...")
    model = apply_lora_to_model(model, {
        'rank': lora_config['rank'],
        'alpha': lora_config['alpha'],
        'dropout': 0.0,  # No dropout for inference
        'target_modules': lora_config['target_modules'],
        'apply_to_backbone': True,
        'apply_to_decoder': True,
    })

    # Load LoRA weights
    print("Loading LoRA weights...")
    load_lora_state_dict(model, checkpoint['lora_state_dict'])

    # Set to eval mode
    model.eval()
    pipeline.model = model

    # Generate audio
    print("\nGenerating audio...")
    print(f"  Tags: {TEST_TAGS}")
    print(f"  Duration: {MAX_DURATION_SEC} seconds")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Top-k: {TOP_K}")
    print(f"  CFG scale: {CFG_SCALE}")

    # Run inference manually to avoid torchaudio.save() in postprocess
    with torch.no_grad():
        # Preprocess inputs
        preprocessed = pipeline.preprocess(
            {"tags": TEST_TAGS, "lyrics": TEST_LYRICS},
            cfg_scale=CFG_SCALE
        )

        # Generate audio
        result = pipeline.forward(
            preprocessed,
            max_audio_length_ms=MAX_DURATION_SEC * 1000,
            temperature=TEMPERATURE,
            topk=TOP_K,
            cfg_scale=CFG_SCALE,
        )

    # Manually save audio using scipy (avoiding torchaudio/torchcodec issues)
    print("\nSaving audio...")
    wav = result["wav"]

    # Convert to numpy
    if isinstance(wav, torch.Tensor):
        wav_np = wav.cpu().numpy()
    else:
        wav_np = wav

    # Ensure 1D or get first channel
    if wav_np.ndim == 2:
        wav_np = wav_np[0]

    # Normalize to int16 range
    wav_np = np.clip(wav_np, -1.0, 1.0)
    wav_int16 = (wav_np * 32767).astype(np.int16)

    # Save using scipy
    wavfile.write(str(OUTPUT_PATH), 48000, wav_int16)

    print(f"\nAudio saved to: {OUTPUT_PATH}")
    print("Inference complete!")


def generate_without_lora():
    """Generate audio without LoRA for comparison."""

    print("=" * 60)
    print("HeartMuLa Baseline Inference (no LoRA)")
    print("=" * 60)

    output_path = SCRIPT_DIR / "output_baseline.wav"

    # Load base model
    print("\nLoading HeartMuLa model...")

    if not check_models_exist("3B"):
        print("Model not found, downloading...")
        download_models_if_needed("3B")

    models_dir = get_models_directory()

    pipeline = HeartMuLaGenPipeline.from_pretrained(
        str(models_dir),
        device=torch.device(DEVICE),
        dtype=DTYPE,
        version="3B",
    )

    # Generate manually to avoid torchaudio.save() in postprocess
    print("\nGenerating baseline audio...")

    with torch.no_grad():
        # Preprocess inputs
        preprocessed = pipeline.preprocess(
            {"tags": TEST_TAGS, "lyrics": TEST_LYRICS},
            cfg_scale=CFG_SCALE
        )

        # Generate audio
        result = pipeline.forward(
            preprocessed,
            max_audio_length_ms=MAX_DURATION_SEC * 1000,
            temperature=TEMPERATURE,
            topk=TOP_K,
            cfg_scale=CFG_SCALE,
        )

    # Manually save audio using scipy (avoiding torchaudio/torchcodec issues)
    print("\nSaving audio...")
    wav = result["wav"]

    # Convert to numpy
    if isinstance(wav, torch.Tensor):
        wav_np = wav.cpu().numpy()
    else:
        wav_np = wav

    # Ensure 1D or get first channel
    if wav_np.ndim == 2:
        wav_np = wav_np[0]

    # Normalize to int16 range
    wav_np = np.clip(wav_np, -1.0, 1.0)
    wav_int16 = (wav_np * 32767).astype(np.int16)

    # Save using scipy
    wavfile.write(str(output_path), 48000, wav_int16)

    print(f"\nBaseline audio saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="Generate baseline without LoRA")
    args = parser.parse_args()

    if args.baseline:
        generate_without_lora()
    else:
        main()
