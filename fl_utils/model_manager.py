"""
Model loading and caching for FL HeartMuLa.
Handles downloading, caching, and loading of HeartMuLa models.
"""

import gc
import os
import sys
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional, Callable

import torch

# Add heartlib to path
_PACKAGE_ROOT = Path(__file__).parent.parent
_FL_UTILS_DIR = Path(__file__).parent

if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

# Import paths module explicitly to avoid relative import issues
def _import_paths():
    module_path = _FL_UTILS_DIR / "paths.py"
    spec = importlib.util.spec_from_file_location("heartmula_paths", str(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_paths = _import_paths()
get_models_directory = _paths.get_models_directory

# Global model cache
_MODEL_CACHE: Dict[str, Any] = {}

# Model variant configurations
MODEL_VARIANTS = {
    "3B": {
        "hf_repos": {
            "model": "HeartMuLa/HeartMuLa-oss-3B",
            "codec": "HeartMuLa/HeartCodec-oss",
            "config": "HeartMuLa/HeartMuLaGen",
        },
        "description": "3B parameter model - Good balance of quality and speed",
        "vram_fp16": 12,
        "vram_4bit": 6,
        "max_duration_ms": 240000,
        "languages": ["en", "zh", "ja", "ko", "es"],
    },
    "7B": {
        "hf_repos": {
            "model": "HeartMuLa/HeartMuLa-oss-7B",
            "codec": "HeartMuLa/HeartCodec-oss",
            "config": "HeartMuLa/HeartMuLaGen",
        },
        "description": "7B parameter model - Higher quality (coming soon)",
        "vram_fp16": 24,
        "vram_4bit": 12,
        "max_duration_ms": 240000,
        "languages": ["en", "zh", "ja", "ko", "es"],
    },
}


def get_variant_list() -> list:
    """Get list of available model variants."""
    return list(MODEL_VARIANTS.keys())


def get_variant_info(variant: str) -> dict:
    """Get info for a specific variant."""
    return MODEL_VARIANTS.get(variant, MODEL_VARIANTS["3B"])


def check_models_exist(variant: str) -> bool:
    """
    Check if model files exist locally.

    Args:
        variant: Model variant (3B or 7B)

    Returns:
        True if all required files exist
    """
    models_dir = get_models_directory()

    # Check for required directories/files
    required = [
        models_dir / f"HeartMuLa-oss-{variant}",
        models_dir / "HeartCodec-oss",
        models_dir / "tokenizer.json",
        models_dir / "gen_config.json",
    ]

    for path in required:
        if not path.exists():
            return False

    return True


def download_models_if_needed(
    variant: str,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Path:
    """
    Download model files from HuggingFace if not present.

    Args:
        variant: Model variant (3B or 7B)
        progress_callback: Optional callback for progress updates

    Returns:
        Path to models directory
    """
    from huggingface_hub import snapshot_download

    models_dir = get_models_directory()
    variant_info = get_variant_info(variant)
    hf_repos = variant_info["hf_repos"]

    total_steps = 3
    current_step = 0

    # Download main model
    model_path = models_dir / f"HeartMuLa-oss-{variant}"
    if not model_path.exists():
        print(f"[FL HeartMuLa] Downloading HeartMuLa-oss-{variant}...")
        snapshot_download(
            repo_id=hf_repos["model"],
            local_dir=str(model_path),
            local_dir_use_symlinks=False,
        )
    current_step += 1
    if progress_callback:
        progress_callback(current_step, total_steps)

    # Download codec
    codec_path = models_dir / "HeartCodec-oss"
    if not codec_path.exists():
        print("[FL HeartMuLa] Downloading HeartCodec-oss...")
        snapshot_download(
            repo_id=hf_repos["codec"],
            local_dir=str(codec_path),
            local_dir_use_symlinks=False,
        )
    current_step += 1
    if progress_callback:
        progress_callback(current_step, total_steps)

    # Download config files (tokenizer.json, gen_config.json)
    tokenizer_path = models_dir / "tokenizer.json"
    gen_config_path = models_dir / "gen_config.json"
    if not tokenizer_path.exists() or not gen_config_path.exists():
        print("[FL HeartMuLa] Downloading config files...")
        snapshot_download(
            repo_id=hf_repos["config"],
            local_dir=str(models_dir),
            local_dir_use_symlinks=False,
            allow_patterns=["tokenizer.json", "gen_config.json"],
        )
    current_step += 1
    if progress_callback:
        progress_callback(current_step, total_steps)

    return models_dir


def load_model(
    variant: str = "3B",
    precision: str = "auto",
    use_4bit: bool = False,
    force_reload: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """
    Load HeartMuLaGenPipeline with caching.

    Args:
        variant: Model variant (3B or 7B)
        precision: Precision mode (auto, fp32, fp16, bf16)
        use_4bit: Whether to use 4-bit quantization
        force_reload: Force reload even if cached
        progress_callback: Optional callback for progress updates

    Returns:
        Model info dict containing pipeline and metadata
    """
    # Determine device (prioritize CUDA, then MPS, then CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Determine dtype
    if precision == "auto":
        if device.type == "cuda":
            dtype = torch.float16
        elif device.type == "mps":
            # MPS works best with float32, fp16 can cause issues
            dtype = torch.float32
        else:
            dtype = torch.float32
    elif precision == "fp32":
        dtype = torch.float32
    elif precision == "fp16":
        dtype = torch.float16
    elif precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    # Create cache key
    cache_key = f"{variant}_{device}_{dtype}_{use_4bit}"

    # Return cached model if available
    if not force_reload and cache_key in _MODEL_CACHE:
        print(f"[FL HeartMuLa] Using cached model: {variant}")
        return _MODEL_CACHE[cache_key]

    # Download models if needed
    print(f"[FL HeartMuLa] Checking model files...")
    models_dir = download_models_if_needed(variant, progress_callback)

    # Import HeartMuLa pipeline
    from heartlib import HeartMuLaGenPipeline

    # Setup quantization config if needed
    bnb_config = None
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            print("[FL HeartMuLa] Using 4-bit quantization")
        except ImportError:
            print("[FL HeartMuLa] WARNING: bitsandbytes not installed, using full precision")
            bnb_config = None

    # Load the pipeline
    print(f"[FL HeartMuLa] Loading HeartMuLa-{variant} pipeline...")
    pipeline = HeartMuLaGenPipeline.from_pretrained(
        pretrained_path=str(models_dir),
        device=device,
        dtype=dtype,
        version=variant,
        bnb_config=bnb_config,
    )

    # Create model info dict
    model_info = {
        "pipeline": pipeline,
        "version": variant,
        "device": device,
        "dtype": dtype,
        "sample_rate": 48000,
        "max_duration_ms": MODEL_VARIANTS[variant]["max_duration_ms"],
        "use_4bit": use_4bit,
    }

    # Cache the model
    _MODEL_CACHE[cache_key] = model_info
    print(f"[FL HeartMuLa] Model loaded successfully!")

    return model_info


def clear_model_cache():
    """Clear the model cache and free GPU memory."""
    global _MODEL_CACHE

    for key in list(_MODEL_CACHE.keys()):
        model_info = _MODEL_CACHE.pop(key)
        if "pipeline" in model_info:
            del model_info["pipeline"]

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print("[FL HeartMuLa] Model cache cleared")


def get_cache_info() -> dict:
    """Get information about cached models."""
    return {
        "cached_models": list(_MODEL_CACHE.keys()),
        "num_cached": len(_MODEL_CACHE),
    }
