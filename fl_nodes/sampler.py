"""
FL HeartMuLa Sampler Node.
Generates audio tokens from conditioning using autoregressive sampling.
"""

import os
import sys
import importlib.util
import gc
import logging
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

from comfy.utils import ProgressBar

# Suppress torchtune KV cache warning (harmless, happens on re-runs with cached model)
# torchtune uses logger.warning(), not warnings.warn()
logging.getLogger("torchtune.modules.attention").setLevel(logging.ERROR)
logging.getLogger("torchtune.modules._export.attention").setLevel(logging.ERROR)
logging.getLogger("torchtune.models.gemma2._attention").setLevel(logging.ERROR)

# Get the package root directory
_PACKAGE_ROOT = os.path.dirname(os.path.dirname(__file__))


class FL_HeartMuLa_Sampler:
    """
    Generate audio tokens from conditioning.

    This node performs the autoregressive generation loop, producing
    audio token frames that can be decoded into a waveform.
    """

    RETURN_TYPES = ("HEARTMULA_TOKENS",)
    RETURN_NAMES = ("audio_tokens",)
    FUNCTION = "sample"
    CATEGORY = "FL HeartMuLa"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "HEARTMULA_MODEL",
                    {
                        "tooltip": "Loaded HeartMuLa model from Model Loader node"
                    }
                ),
                "conditioning": (
                    "HEARTMULA_COND",
                    {
                        "tooltip": "Conditioning from the Conditioning node"
                    }
                ),
            },
            "optional": {
                "max_duration_sec": (
                    "INT",
                    {
                        "default": 60,
                        "min": 10,
                        "max": 240,
                        "step": 10,
                        "tooltip": "Maximum audio duration in seconds (max 240s / 4 minutes)"
                    }
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Sampling temperature (higher = more random/creative)"
                    }
                ),
                "top_k": (
                    "INT",
                    {
                        "default": 50,
                        "min": 1,
                        "max": 500,
                        "step": 10,
                        "tooltip": "Top-k sampling (lower = more focused/consistent)"
                    }
                ),
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 2147483647,
                        "tooltip": "Random seed (-1 for random)"
                    }
                ),
            }
        }

    def sample(
        self,
        model: dict,
        conditioning: dict,
        max_duration_sec: int = 60,
        temperature: float = 1.0,
        top_k: int = 50,
        seed: int = -1,
    ) -> Tuple[dict]:
        """
        Generate audio tokens from conditioning.

        Args:
            model: Loaded model info dict
            conditioning: Conditioning dict from Conditioning node
            max_duration_sec: Maximum duration in seconds
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            seed: Random seed

        Returns:
            Tuple containing audio tokens dict
        """
        # Handle seed
        if seed == -1:
            seed = int(np.random.randint(0, 2147483647))

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # MPS doesn't have manual_seed, it uses torch.manual_seed

        print(f"\n{'='*60}")
        print(f"[FL HeartMuLa] Sampling Audio Tokens")
        print(f"{'='*60}")
        print(f"Max Duration: {max_duration_sec}s")
        print(f"Temperature: {temperature}")
        print(f"Top-K: {top_k}")
        print(f"Seed: {seed}")

        pipeline = model["pipeline"]
        device = model["device"]
        dtype = model["dtype"]
        max_duration_ms = model.get("max_duration_ms", 240000)

        # Clamp duration
        max_duration_ms = min(max_duration_sec * 1000, max_duration_ms)
        max_audio_frames = max_duration_ms // 80

        # Extract conditioning tensors and move to device
        prompt_tokens = conditioning["tokens"].to(device)
        prompt_tokens_mask = conditioning["tokens_mask"].to(device)
        continuous_segment = conditioning["muq_embed"].to(device)
        starts = conditioning["muq_idx"]
        prompt_pos = conditioning["pos"].to(device)
        cfg_scale = conditioning["cfg_scale"]

        print(f"CFG Scale: {cfg_scale}")
        print(f"Max frames: {max_audio_frames}")
        print(f"{'='*60}\n")

        # Setup progress bar
        pbar = ProgressBar(max_audio_frames)

        frames = []
        bs_size = 2 if cfg_scale != 1.0 else 1

        try:
            # Reset and setup model caches (reset first to avoid warnings on re-runs)
            try:
                pipeline.model.reset_caches()
            except (RuntimeError, AttributeError):
                pass  # Caches may not exist yet on first run
            pipeline.model.setup_caches(bs_size)

            # Generate first frame from prompt
            with torch.autocast(device_type=device.type, dtype=dtype):
                curr_token = pipeline.model.generate_frame(
                    tokens=prompt_tokens,
                    tokens_mask=prompt_tokens_mask,
                    input_pos=prompt_pos,
                    temperature=temperature,
                    topk=top_k,
                    cfg_scale=cfg_scale,
                    continuous_segments=continuous_segment,
                    starts=starts,
                )
            frames.append(curr_token[0:1,])

            # Helper to pad audio tokens
            def _pad_audio_token(token: torch.Tensor):
                padded_token = (
                    torch.ones(
                        (token.shape[0], pipeline._parallel_number),
                        device=token.device,
                        dtype=torch.long,
                    )
                    * pipeline.config.empty_id
                )
                padded_token[:, :-1] = token
                padded_token = padded_token.unsqueeze(1)
                padded_token_mask = torch.ones_like(
                    padded_token, device=token.device, dtype=torch.bool
                )
                padded_token_mask[..., -1] = False
                return padded_token, padded_token_mask

            # Autoregressive generation loop
            for i in tqdm(range(max_audio_frames), desc="Generating audio frames"):
                curr_token, curr_token_mask = _pad_audio_token(curr_token)

                with torch.autocast(device_type=device.type, dtype=dtype):
                    curr_token = pipeline.model.generate_frame(
                        tokens=curr_token,
                        tokens_mask=curr_token_mask,
                        input_pos=(prompt_pos[..., -1:] + i + 1).to(device),
                        temperature=temperature,
                        topk=top_k,
                        cfg_scale=cfg_scale,
                        continuous_segments=None,
                        starts=None,
                    )

                # Update progress
                pbar.update_absolute(i + 1)

                # Check for EOS token
                if torch.any(curr_token[0:1, :] >= pipeline.config.audio_eos_id):
                    print(f"[FL HeartMuLa] EOS detected at frame {i + 1}")
                    pbar.update_absolute(max_audio_frames)
                    break

                frames.append(curr_token[0:1,])

            # Stack frames: [num_frames, 1, codebooks] -> [codebooks, num_frames]
            frames_tensor = torch.stack(frames).permute(1, 2, 0).squeeze(0)

            print(f"\n{'='*60}")
            print(f"[FL HeartMuLa] Sampling Complete!")
            print(f"Generated {len(frames)} frames ({len(frames) * 80 / 1000:.2f}s)")
            print(f"Token shape: {frames_tensor.shape}")
            print(f"{'='*60}\n")

            # Build output dict
            audio_tokens = {
                "frames": frames_tensor,
                "num_frames": len(frames),
                "sample_rate": model["sample_rate"],
                "seed": seed,
            }

            return (audio_tokens,)

        finally:
            # Clean up
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()