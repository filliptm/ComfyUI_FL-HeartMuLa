"""
FL HeartMuLa Decode Node.
Decodes audio tokens into waveform using HeartCodec.
"""

import os
import sys
import importlib.util
import gc
from typing import Tuple

import torch

from comfy.utils import ProgressBar

# Get the package root directory
_PACKAGE_ROOT = os.path.dirname(os.path.dirname(__file__))

# Import modules explicitly from our package to avoid conflicts
def _import_from_package(module_name, file_name):
    """Import a module from our package specifically."""
    module_path = os.path.join(_PACKAGE_ROOT, "fl_utils", f"{file_name}.py")
    spec = importlib.util.spec_from_file_location(f"heartmula_{module_name}", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import audio utilities
_audio_utils = _import_from_package("audio_utils", "audio_utils")
tensor_to_comfyui_audio = _audio_utils.tensor_to_comfyui_audio
empty_audio = _audio_utils.empty_audio


class FL_HeartMuLa_Decode:
    """
    Decode audio tokens into waveform.

    This node uses HeartCodec to convert the generated audio tokens
    from the Sampler into an actual audio waveform.
    """

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "decode"
    CATEGORY = "FL HeartMuLa"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "HEARTMULA_MODEL",
                    {
                        "tooltip": "Loaded HeartMuLa model (contains HeartCodec)"
                    }
                ),
                "audio_tokens": (
                    "HEARTMULA_TOKENS",
                    {
                        "tooltip": "Audio tokens from the Sampler node"
                    }
                ),
            },
        }

    def decode(
        self,
        model: dict,
        audio_tokens: dict,
    ) -> Tuple[dict]:
        """
        Decode audio tokens into waveform.

        Args:
            model: Loaded model info dict (contains HeartCodec)
            audio_tokens: Audio tokens dict from Sampler

        Returns:
            Tuple containing ComfyUI AUDIO dict
        """
        print(f"\n{'='*60}")
        print(f"[FL HeartMuLa] Decoding Audio")
        print(f"{'='*60}")

        pipeline = model["pipeline"]
        sample_rate = model["sample_rate"]
        ultra_low_mem = model.get("ultra_low_mem", False)

        frames = audio_tokens["frames"]
        num_frames = audio_tokens["num_frames"]

        print(f"Input frames: {frames.shape}")
        print(f"Frame count: {num_frames}")

        # Pre-cleanup for ultra low memory mode
        if ultra_low_mem:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Progress bar for decode operation
        pbar = ProgressBar(2)
        pbar.update_absolute(0)

        try:
            # Decode using HeartCodec
            pbar.update_absolute(1)  # Show we're working
            with torch.no_grad():
                wav = pipeline.audio_codec.detokenize(frames)
            pbar.update_absolute(2)  # Complete

            print(f"Output waveform: {wav.shape}")
            print(f"Duration: {wav.shape[-1] / sample_rate:.2f}s")
            print(f"Sample rate: {sample_rate}Hz")
            print(f"{'='*60}\n")

            # Convert to ComfyUI audio format
            audio = tensor_to_comfyui_audio(wav.cpu(), sample_rate)

            return (audio,)

        except Exception as e:
            print(f"[FL HeartMuLa] ERROR: Decoding failed!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

            return (empty_audio(sample_rate),)

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
