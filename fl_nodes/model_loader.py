"""
FL HeartMuLa Model Loader Node.
Loads the HeartMuLa music generation model with configurable options.
"""

import os
import sys
import importlib.util
from typing import Tuple

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

# Import our model_manager module
_model_manager = _import_from_package("model_manager", "model_manager")

load_model = _model_manager.load_model
get_variant_list = _model_manager.get_variant_list
get_variant_info = _model_manager.get_variant_info
MODEL_VARIANTS = _model_manager.MODEL_VARIANTS
clear_model_cache = _model_manager.clear_model_cache
get_recommended_memory_mode = _model_manager.get_recommended_memory_mode
get_available_vram_gb = _model_manager.get_available_vram_gb

# Memory mode options
MEMORY_MODES = ["auto", "normal", "low", "ultra"]


class FL_HeartMuLa_ModelLoader:
    """
    Load HeartMuLa music generation model with variant selection and options.

    This node loads the HeartMuLa AI music generation model. The 3B model offers
    a good balance of quality and speed. 4-bit quantization can reduce VRAM usage.
    """

    RETURN_TYPES = ("HEARTMULA_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "🎵FL HeartMuLa"

    @classmethod
    def INPUT_TYPES(cls):
        variants = get_variant_list()
        return {
            "required": {
                "model_version": (
                    variants,
                    {
                        "default": "3B",
                        "tooltip": "Model size. 3B is released and recommended."
                    }
                ),
            },
            "optional": {
                "memory_mode": (
                    MEMORY_MODES,
                    {
                        "default": "auto",
                        "tooltip": "Memory mode: auto (recommended), normal (fast, high VRAM), low (slower, less VRAM), ultra (minimum VRAM)"
                    }
                ),
                "precision": (
                    ["auto", "fp32", "fp16", "bf16"],
                    {
                        "default": "auto",
                        "tooltip": "Model precision. Auto uses fp16 on CUDA, fp32 on CPU."
                    }
                ),
                "use_4bit": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Use 4-bit quantization to reduce VRAM (requires bitsandbytes)"
                    }
                ),
                "force_reload": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Force reload model even if already cached"
                    }
                ),
            }
        }

    def load_model(
        self,
        model_version: str,
        memory_mode: str = "auto",
        precision: str = "auto",
        use_4bit: bool = False,
        force_reload: bool = False
    ) -> Tuple[dict]:
        """
        Load the HeartMuLa model.

        Args:
            model_version: Which model variant to load (3B or 7B)
            memory_mode: Memory mode - "auto", "normal", "low", or "ultra"
            precision: Model precision mode
            use_4bit: Enable 4-bit quantization
            force_reload: Force reload even if cached

        Returns:
            Tuple containing the model info dict
        """
        # Check for unreleased models
        if model_version == "7B":
            raise ValueError(
                "\n" + "="*60 + "\n"
                "[FL HeartMuLa] 7B Model Coming Soon!\n"
                "="*60 + "\n"
                "The 7B parameter model has not been released yet.\n"
                "Please use the 3B model for now - it offers excellent quality!\n\n"
                "Stay tuned for updates on the 7B release.\n"
                + "="*60
            )

        # Resolve memory mode
        if memory_mode == "auto":
            resolved_mode = get_recommended_memory_mode(model_version, use_4bit)
            available_vram = get_available_vram_gb()
            print(f"[FL HeartMuLa] Auto-detected memory mode: {resolved_mode} (available VRAM: {available_vram:.1f}GB)")
        else:
            resolved_mode = memory_mode

        # Get variant info for logging
        variant_info = get_variant_info(model_version)

        print(f"\n{'='*60}")
        print(f"[FL HeartMuLa] Loading Model")
        print(f"{'='*60}")
        print(f"Version: {model_version}")
        print(f"Description: {variant_info['description']}")
        print(f"Max Duration: {variant_info['max_duration_ms'] // 1000}s")
        print(f"Languages: {', '.join(variant_info['languages'])}")
        print(f"Memory Mode: {resolved_mode}")
        vram = variant_info['vram_4bit'] if use_4bit else variant_info['vram_fp16']
        if resolved_mode == "ultra":
            print(f"VRAM Required: ~{max(6, vram - 6)}GB (ultra low memory mode)")
        elif resolved_mode == "low":
            print(f"VRAM Required: ~{max(8, vram - 4)}GB (low memory mode)")
        else:
            print(f"VRAM Required: ~{vram}GB")
        print(f"Precision: {precision}")
        print(f"4-bit Quantization: {use_4bit}")
        print(f"{'='*60}\n")

        try:
            # Progress bar for loading stages
            pbar = ProgressBar(4)

            def progress_callback(current, total):
                pbar.update_absolute(current)

            model_info = load_model(
                variant=model_version,
                precision=precision,
                use_4bit=use_4bit,
                force_reload=force_reload,
                progress_callback=progress_callback,
            )

            # Add memory mode flags to model_info for sampler
            model_info["memory_mode"] = resolved_mode
            model_info["low_mem"] = resolved_mode in ("low", "ultra")
            model_info["ultra_low_mem"] = resolved_mode == "ultra"

            pbar.update_absolute(4)
            print(f"[FL HeartMuLa] Model loaded successfully!")
            return (model_info,)

        except FileNotFoundError as e:
            print(f"\n{'='*60}")
            print(f"[FL HeartMuLa] ERROR: Model files not found!")
            print(f"{'='*60}")
            print(f"The model will be downloaded automatically from HuggingFace.")
            print(f"Expected location: ComfyUI/models/heartmula/")
            print(f"{'='*60}\n")
            raise e

        except Exception as e:
            print(f"[FL HeartMuLa] ERROR loading model: {e}")
            raise e
