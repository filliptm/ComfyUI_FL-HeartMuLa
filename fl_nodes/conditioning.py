"""
FL HeartMuLa Conditioning Node.
Tokenizes lyrics and tags into conditioning tensors for generation.
"""

import os
import sys
import importlib.util
from typing import Tuple

import torch

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


class FL_HeartMuLa_Conditioning:
    """
    Create conditioning from lyrics and tags for HeartMuLa generation.

    This node tokenizes the input text and prepares the prompt tensors
    that will be used by the Sampler node for music generation.
    """

    RETURN_TYPES = ("HEARTMULA_COND",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "create_conditioning"
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
                "lyrics": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "[Verse]\nHello world, this is a test\nSinging with AI today\n[Chorus]\nMusic generation is here\nMaking melodies so clear",
                        "tooltip": "Lyrics with section markers like [Verse], [Chorus], [Bridge], [Intro], [Outro]"
                    }
                ),
                "tags": (
                    "STRING",
                    {
                        "default": "pop, female vocal, energetic",
                        "tooltip": "Comma-separated style tags (genre, vocal type, mood, instruments)"
                    }
                ),
            },
            "optional": {
                "cfg_scale": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 1.0,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Classifier-free guidance strength (1.0 = no CFG)"
                    }
                ),
            }
        }

    def create_conditioning(
        self,
        model: dict,
        lyrics: str,
        tags: str,
        cfg_scale: float = 1.5,
    ) -> Tuple[dict]:
        """
        Create conditioning tensors from lyrics and tags.

        Args:
            model: Loaded model info dict from Model Loader
            lyrics: Formatted lyrics with section markers
            tags: Comma-separated style tags
            cfg_scale: CFG scale (affects tensor batching)

        Returns:
            Tuple containing conditioning dict
        """
        print(f"\n{'='*60}")
        print(f"[FL HeartMuLa] Creating Conditioning")
        print(f"{'='*60}")
        print(f"Tags: {tags[:80]}{'...' if len(tags) > 80 else ''}")
        print(f"Lyrics preview: {lyrics[:100]}{'...' if len(lyrics) > 100 else ''}")
        print(f"CFG Scale: {cfg_scale}")

        pipeline = model["pipeline"]
        device = model["device"]
        dtype = model["dtype"]

        # Process tags - wrap with special tokens
        tags_processed = tags.lower()
        if not tags_processed.startswith("<tag>"):
            tags_processed = f"<tag>{tags_processed}"
        if not tags_processed.endswith("</tag>"):
            tags_processed = f"{tags_processed}</tag>"

        # Tokenize tags
        tags_ids = pipeline.text_tokenizer.encode(tags_processed).ids
        if tags_ids[0] != pipeline.config.text_bos_id:
            tags_ids = [pipeline.config.text_bos_id] + tags_ids
        if tags_ids[-1] != pipeline.config.text_eos_id:
            tags_ids = tags_ids + [pipeline.config.text_eos_id]

        # MuQ embedding placeholder (for future reference audio support)
        muq_embed = torch.zeros((pipeline._muq_dim,), dtype=dtype)
        muq_idx = len(tags_ids)

        # Process lyrics
        lyrics_processed = lyrics.lower()
        lyrics_ids = pipeline.text_tokenizer.encode(lyrics_processed).ids
        if lyrics_ids[0] != pipeline.config.text_bos_id:
            lyrics_ids = [pipeline.config.text_bos_id] + lyrics_ids
        if lyrics_ids[-1] != pipeline.config.text_eos_id:
            lyrics_ids = lyrics_ids + [pipeline.config.text_eos_id]

        # Build prompt tensors: [tags, muq_placeholder, lyrics]
        prompt_len = len(tags_ids) + 1 + len(lyrics_ids)
        parallel_number = pipeline._parallel_number

        tokens = torch.zeros([prompt_len, parallel_number], dtype=torch.long)
        tokens[:len(tags_ids), -1] = torch.tensor(tags_ids)
        tokens[len(tags_ids) + 1:, -1] = torch.tensor(lyrics_ids)

        tokens_mask = torch.zeros_like(tokens, dtype=torch.bool)
        tokens_mask[:, -1] = True

        # Handle CFG batching
        bs_size = 2 if cfg_scale != 1.0 else 1

        def _cfg_cat(tensor: torch.Tensor):
            tensor = tensor.unsqueeze(0)
            if cfg_scale != 1.0:
                tensor = torch.cat([tensor, tensor], dim=0)
            return tensor

        # Build conditioning dict
        conditioning = {
            "tokens": _cfg_cat(tokens).to(device),
            "tokens_mask": _cfg_cat(tokens_mask).to(device),
            "muq_embed": _cfg_cat(muq_embed).to(device),
            "muq_idx": [muq_idx] * bs_size,
            "pos": _cfg_cat(torch.arange(prompt_len, dtype=torch.long)).to(device),
            "cfg_scale": cfg_scale,
            "prompt_len": prompt_len,
        }

        print(f"Prompt length: {prompt_len} tokens")
        print(f"Batch size: {bs_size} (CFG {'enabled' if cfg_scale != 1.0 else 'disabled'})")
        print(f"{'='*60}\n")

        return (conditioning,)
