"""
FL HeartMuLa Transcribe Node.
Transcribe lyrics from audio using HeartTranscriptor.
"""

import os
import sys
import importlib.util
from typing import Tuple

import torch

# Get the package root directory
_PACKAGE_ROOT = os.path.dirname(os.path.dirname(__file__))

# Import modules explicitly from our package
def _import_from_package(module_name, file_name):
    """Import a module from our package specifically."""
    module_path = os.path.join(_PACKAGE_ROOT, "fl_utils", f"{file_name}.py")
    spec = importlib.util.spec_from_file_location(f"heartmula_{module_name}", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_audio_utils = _import_from_package("audio_utils", "audio_utils")
_paths = _import_from_package("paths", "paths")

comfyui_audio_to_tensor = _audio_utils.comfyui_audio_to_tensor
get_models_directory = _paths.get_models_directory


class FL_HeartMuLa_Transcribe:
    """
    Transcribe lyrics from audio using HeartTranscriptor.

    HeartTranscriptor is a Whisper-based model fine-tuned for lyrics transcription.
    It can extract lyrics from music audio, including generated music.
    """

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "transcribe"
    CATEGORY = "🎵FL HeartMuLa"

    # Cache for the transcriptor pipeline
    _transcriptor_cache = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": (
                    "AUDIO",
                    {
                        "tooltip": "Audio to transcribe lyrics from"
                    }
                ),
            },
        }

    def _load_transcriptor(self):
        """Load the HeartTranscriptor model."""
        if FL_HeartMuLa_Transcribe._transcriptor_cache is not None:
            return FL_HeartMuLa_Transcribe._transcriptor_cache

        print("[FL HeartMuLa] Loading HeartTranscriptor...")

        # Add heartlib to path
        if str(_PACKAGE_ROOT) not in sys.path:
            sys.path.insert(0, str(_PACKAGE_ROOT))

        try:
            from heartlib import HeartTranscriptorPipeline

            models_dir = get_models_directory()
            transcriptor_path = models_dir / "HeartTranscriptor-oss"

            # Check if model exists
            if not transcriptor_path.exists():
                print("[FL HeartMuLa] HeartTranscriptor not found, downloading...")
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id="HeartMuLa/HeartTranscriptor-oss",
                    local_dir=str(transcriptor_path),
                    local_dir_use_symlinks=False,
                )

            # Determine device and dtype
            if torch.cuda.is_available():
                device = torch.device("cuda")
                dtype = torch.float16
            else:
                device = torch.device("cpu")
                dtype = torch.float32

            # Load the pipeline with device and dtype
            try:
                pipeline = HeartTranscriptorPipeline.from_pretrained(
                    str(transcriptor_path),
                    device=device,
                    dtype=dtype
                )
            except TypeError:
                # Fallback for older versions that don't require device/dtype
                pipeline = HeartTranscriptorPipeline.from_pretrained(str(transcriptor_path))

            FL_HeartMuLa_Transcribe._transcriptor_cache = pipeline

            print("[FL HeartMuLa] HeartTranscriptor loaded!")
            return pipeline

        except Exception as e:
            print(f"[FL HeartMuLa] ERROR loading HeartTranscriptor: {e}")
            raise

    def transcribe(self, audio: dict) -> Tuple[str]:
        """
        Transcribe lyrics from audio.

        Args:
            audio: ComfyUI AUDIO dict

        Returns:
            Tuple containing transcribed lyrics string
        """
        print(f"\n{'='*60}")
        print(f"[FL HeartMuLa] Transcribing Lyrics")
        print(f"{'='*60}")

        try:
            # Get audio tensor
            waveform, sample_rate = comfyui_audio_to_tensor(audio)
            print(f"Input audio: {waveform.shape}, {sample_rate}Hz")

            # Load transcriptor
            transcriptor = self._load_transcriptor()

            # Prepare audio for transcription
            # HeartTranscriptor expects audio dict similar to ComfyUI format
            audio_input = {
                "waveform": waveform,
                "sample_rate": sample_rate
            }

            # Run transcription
            result = transcriptor(audio_input)

            # Extract text from result
            if isinstance(result, dict):
                lyrics = result.get("text", "")
            elif isinstance(result, str):
                lyrics = result
            else:
                lyrics = str(result)

            print(f"[FL HeartMuLa] Transcription complete!")
            print(f"Transcribed: {lyrics[:200]}{'...' if len(lyrics) > 200 else ''}")
            print(f"{'='*60}\n")

            return (lyrics,)

        except Exception as e:
            print(f"[FL HeartMuLa] ERROR: Transcription failed!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return ("",)
