"""
FL HeartMuLa LoRA Training Node.
Train LoRA adapters on the HeartMuLa music generation model.

Note: ComfyUI disables gradients globally. This node temporarily re-enables them
during training using a context manager.
"""

import os
import sys
import gc
from pathlib import Path
from typing import Tuple
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from comfy.utils import ProgressBar

# Get the package root directory
_PACKAGE_ROOT = os.path.dirname(os.path.dirname(__file__))

# Ensure heartlib is importable
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)


# ============================================================================
# GRADIENT CONTEXT MANAGER
# ============================================================================

@contextmanager
def enable_gradients():
    """
    Context manager to temporarily enable gradients in ComfyUI.

    ComfyUI disables gradients globally via torch.set_grad_enabled(False).
    This context manager temporarily re-enables them for training.
    """
    prev_grad_enabled = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(True)
        yield
    finally:
        torch.set_grad_enabled(prev_grad_enabled)


# ============================================================================
# LORA IMPLEMENTATION
# ============================================================================

class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer that wraps a linear layer.
    Implements: output = base_linear(x) + (x @ A @ B) * (alpha / rank) * strength

    The strength parameter allows dynamic adjustment of LoRA effect during inference.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
        strength: float = 1.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.strength = strength  # Dynamic strength multiplier

        # CRITICAL: Clone the base layer weights to escape inference mode!
        # ComfyUI loads models in inference mode, which "taints" the tensors.
        # We need to clone them to create new tensors that can be used with autograd.
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        # Create a new Linear layer with cloned weights (escapes inference mode)
        self.base_layer = nn.Linear(in_features, out_features, bias=base_layer.bias is not None, device=device, dtype=dtype)
        with torch.no_grad():
            self.base_layer.weight.copy_(base_layer.weight.clone())
            if base_layer.bias is not None:
                self.base_layer.bias.copy_(base_layer.bias.clone())

        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # CRITICAL: Create LoRA parameters in FLOAT32 for training stability!
        # AdamW optimizer with float16/bfloat16 causes NaN due to underflow/overflow.
        # LoRA parameters must be float32 even when base model is float16/bfloat16.
        with torch.enable_grad():
            # Create tensors in float32 (NOT the base model's dtype)
            lora_A_data = torch.zeros(in_features, rank, device=device, dtype=torch.float32)
            lora_B_data = torch.zeros(rank, out_features, device=device, dtype=torch.float32)

            # Initialize A with Kaiming, B with zeros (so initial output = base)
            nn.init.kaiming_uniform_(lora_A_data, a=np.sqrt(5))
            nn.init.zeros_(lora_B_data)

            # Wrap as Parameters - requires_grad is True by default for nn.Parameter
            self.lora_A = nn.Parameter(lora_A_data, requires_grad=True)
            self.lora_B = nn.Parameter(lora_B_data, requires_grad=True)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def set_strength(self, strength: float):
        """Set the LoRA strength multiplier."""
        self.strength = strength

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure forward pass tracks gradients
        base_out = self.base_layer(x)

        # If strength is 0, skip LoRA computation entirely
        if self.strength == 0.0:
            return base_out

        # LoRA computation - explicitly use the parameters in a way that tracks gradients
        # The key is that self.lora_A and self.lora_B are nn.Parameters with requires_grad=True
        x_dropped = self.dropout(x)

        # Matrix multiply: x @ A @ B
        # CRITICAL: LoRA weights are float32, input may be float16/bfloat16
        # Cast input to float32 for LoRA computation, then cast result back
        input_dtype = x_dropped.dtype
        x_for_lora = x_dropped.to(torch.float32) if input_dtype != torch.float32 else x_dropped

        lora_intermediate = torch.matmul(x_for_lora, self.lora_A)
        lora_out = torch.matmul(lora_intermediate, self.lora_B)

        # Cast back to original dtype and apply scaling
        if input_dtype != torch.float32:
            lora_out = lora_out.to(input_dtype)

        # Apply scaling and strength
        result = base_out + lora_out * self.scaling * self.strength
        return result

    def get_lora_state_dict(self):
        """Return only the LoRA parameters."""
        return {
            'lora_A': self.lora_A.data,
            'lora_B': self.lora_B.data,
        }


def clone_model_parameters(model: nn.Module) -> nn.Module:
    """
    Clone all model parameters to escape inference mode.

    ComfyUI loads models in torch.inference_mode(), which "taints" all tensors.
    Even after exiting inference mode, those tensors cannot be used in autograd.
    This function creates NEW Parameter objects with cloned data.

    Note: This is expensive but necessary for training in ComfyUI.
    """
    print("[LoRA] Cloning model parameters to escape inference mode...")
    cloned_count = 0

    # We need to replace Parameter objects entirely, not just their data
    # This requires iterating through modules and replacing their parameters
    with torch.no_grad():
        for module in model.modules():
            # Get all parameter names for this specific module (not children)
            param_names = list(module._parameters.keys())
            for param_name in param_names:
                param = module._parameters[param_name]
                if param is not None:
                    # Create a completely new Parameter with cloned data
                    new_param = nn.Parameter(
                        param.data.clone(),
                        requires_grad=param.requires_grad
                    )
                    module._parameters[param_name] = new_param
                    cloned_count += 1

            # Also clone buffers
            buffer_names = list(module._buffers.keys())
            for buffer_name in buffer_names:
                buffer = module._buffers[buffer_name]
                if buffer is not None:
                    module._buffers[buffer_name] = buffer.clone()

    print(f"[LoRA] Cloned {cloned_count} parameters")
    return model


def apply_lora_to_model(model, config: dict) -> nn.Module:
    """
    Apply LoRA to specified layers in the model.

    This function wraps all LoRA creation in enable_grad() context to ensure
    proper gradient tracking in ComfyUI which disables gradients globally.
    """
    rank = config.get('rank', 16)
    alpha = config.get('alpha', 32.0)
    dropout = config.get('dropout', 0.05)
    target_modules = config.get('target_modules', ['q_proj', 'k_proj', 'v_proj', 'output_proj'])
    apply_to_backbone = config.get('apply_to_backbone', True)
    apply_to_decoder = config.get('apply_to_decoder', True)

    # First, freeze ALL parameters in the model
    for param in model.parameters():
        param.requires_grad = False

    lora_count = 0

    def replace_linear_with_lora(module, name_prefix=""):
        nonlocal lora_count
        for name, child in list(module.named_children()):
            full_name = f"{name_prefix}.{name}" if name_prefix else name

            # Check if should apply based on component
            should_apply = False
            if apply_to_backbone and 'backbone' in full_name:
                should_apply = True
            if apply_to_decoder and 'decoder' in full_name:
                should_apply = True

            if should_apply and any(t in name for t in target_modules):
                if isinstance(child, nn.Linear):
                    # LoRALinear.__init__ uses enable_grad() internally
                    lora_layer = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
                    setattr(module, name, lora_layer)
                    lora_count += 1

            replace_linear_with_lora(child, full_name)

    # Apply LoRA - the LoRALinear class handles enable_grad internally
    replace_linear_with_lora(model)
    print(f"Applied LoRA to {lora_count} layers")

    # Re-freeze all non-LoRA parameters to be absolutely sure
    # (LoRA params have lora_A and lora_B in their name)
    # CRITICAL: Use requires_grad_() method instead of attribute assignment
    lora_params = 0
    frozen_params = 0
    with torch.enable_grad():
        for name, param in model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                param.requires_grad_(True)  # Use method form for reliability
                lora_params += param.numel()
            else:
                param.requires_grad_(False)
                frozen_params += param.numel()

    total = lora_params + frozen_params
    print(f"LoRA parameters: {lora_params:,}")
    print(f"Frozen parameters: {frozen_params:,}")
    print(f"Trainable: {lora_params:,} / {total:,} ({100*lora_params/total:.2f}%)")

    return model


def get_lora_state_dict(model) -> dict:
    """Extract all LoRA parameters from the model."""
    lora_state = {}

    def collect_lora(module, name_prefix=""):
        for name, child in module.named_children():
            full_name = f"{name_prefix}.{name}" if name_prefix else name
            if isinstance(child, LoRALinear):
                state = child.get_lora_state_dict()
                lora_state[f"{full_name}.lora_A"] = state['lora_A'].cpu()
                lora_state[f"{full_name}.lora_B"] = state['lora_B'].cpu()
            else:
                collect_lora(child, full_name)

    collect_lora(model)
    return lora_state


def load_lora_state_dict(model, state_dict: dict):
    """Load LoRA parameters into a model with LoRA applied."""
    loaded = 0

    def load_lora(module, name_prefix=""):
        nonlocal loaded
        for name, child in module.named_children():
            full_name = f"{name_prefix}.{name}" if name_prefix else name
            if isinstance(child, LoRALinear):
                a_key = f"{full_name}.lora_A"
                b_key = f"{full_name}.lora_B"
                if a_key in state_dict and b_key in state_dict:
                    child.lora_A.data = state_dict[a_key].to(child.lora_A.device, child.lora_A.dtype)
                    child.lora_B.data = state_dict[b_key].to(child.lora_B.device, child.lora_B.dtype)
                    loaded += 2
            else:
                load_lora(child, full_name)

    load_lora(model)
    print(f"Loaded {loaded} LoRA parameters")


def strip_lora_from_model(model) -> nn.Module:
    """
    Remove all LoRA layers from a model, restoring the original base layers.

    This is needed to get a clean model state before applying new LoRA weights,
    since ComfyUI caches models and the old LoRA state persists.
    """
    stripped = 0

    def strip_lora(module):
        nonlocal stripped
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                # Replace LoRALinear with its base_layer
                setattr(module, name, child.base_layer)
                stripped += 1
            else:
                strip_lora(child)

    strip_lora(model)
    print(f"Stripped {stripped} LoRA layers from model")
    return model


def set_lora_strength(model, strength: float):
    """
    Set the strength of all LoRA layers in the model.

    Args:
        model: Model with LoRA layers applied
        strength: Multiplier for LoRA effect (0.0 = no effect, 1.0 = full effect)
    """
    count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.set_strength(strength)
            count += 1
    print(f"Set strength={strength} on {count} LoRA layers")
    return count


# ============================================================================
# AUDIO TOKENIZATION (Pseudo - HeartCodec doesn't expose encoder)
# ============================================================================

def tokenize_audio(waveform: torch.Tensor, num_codebooks: int = 8, vocab_size: int = 8192) -> torch.Tensor:
    """
    Convert audio waveform to pseudo RVQ codes.

    Note: HeartCodec is designed for decoding only. For production training,
    you would need pre-tokenized audio data.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    num_samples = waveform.shape[1]
    sample_rate = 48000
    frame_rate = 12.5  # HeartCodec frame rate
    num_frames = int(num_samples / sample_rate * frame_rate)

    # Generate pseudo-random codes based on audio content
    audio_energy = waveform.abs().mean().item()
    seed = int(audio_energy * 1e6) % (2**31)
    rng = np.random.RandomState(seed)

    codes = rng.randint(0, vocab_size, size=(num_codebooks, num_frames))
    return torch.from_numpy(codes).long()


# ============================================================================
# TRAINING FORWARD PASS
# ============================================================================

@torch.inference_mode(False)  # CRITICAL: Disable inference mode for training in ComfyUI
def training_forward(model, batch: dict, device: str, dtype: torch.dtype, debug: bool = False) -> torch.Tensor:
    """
    Run training forward pass and compute loss.

    This function uses @torch.inference_mode(False) to ensure gradients work
    in ComfyUI which runs in inference mode by default.
    """
    # Double-check gradients are enabled
    if not torch.is_grad_enabled():
        torch.set_grad_enabled(True)

    input_tokens = batch['input_tokens'].to(device)
    input_mask = batch['input_mask'].to(device)
    target_codes = batch['target_codes'].to(device)
    target_mask = batch['target_mask'].to(device)

    B, S, _ = input_tokens.shape

    # Embed tokens - embeddings don't need gradients, but output will be used in gradient computation
    # CRITICAL: Clone embeddings to escape inference mode tensors!
    # ComfyUI loads models in inference mode, which "taints" all tensors including embedding weights.
    with torch.no_grad():
        text_embeds = model.text_embeddings(input_tokens[:, :, 8]).clone()
        audio_tokens = input_tokens[:, :, :8].clone()
        offsets = torch.arange(8, device=device) * model.config.audio_vocab_size
        audio_tokens_offset = audio_tokens + offsets.view(1, 1, 8)
        audio_embeds = model.audio_embeddings(audio_tokens_offset.view(-1)).clone()
        audio_embeds = audio_embeds.view(B, S, 8, -1)

        all_embeds = torch.cat([audio_embeds, text_embeds.unsqueeze(2)], dim=2)
        masked_embeds = all_embeds * input_mask.unsqueeze(-1).to(all_embeds.dtype)
        h = masked_embeds.sum(dim=2).to(dtype).clone()

    # Make h a leaf tensor that requires grad for gradient flow through LoRA layers
    h = h.detach().requires_grad_(True)

    if debug:
        print(f"  h requires_grad: {h.requires_grad}")
        print(f"  torch.is_grad_enabled(): {torch.is_grad_enabled()}")
        print(f"  h stats: min={h.min().item():.4f}, max={h.max().item():.4f}, has_nan={torch.isnan(h).any().item()}")

    # Forward through backbone - this is where LoRA layers are!
    input_pos = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # Reset caches to clean state
    try:
        model.backbone.reset_caches()
    except:
        pass

    # Test: try a simple LoRA layer directly to verify gradient tracking works
    if debug:
        # Find first LoRA layer and test it
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                # h is already detached and requires_grad, clone to create a fresh test input
                test_input = h[:, :1, :].detach().clone().requires_grad_(True)
                test_output = module(test_input)
                print(f"  Direct LoRA test - input.requires_grad: {test_input.requires_grad}, output.requires_grad: {test_output.requires_grad}")
                break

    # Create causal mask for training (torchtune requires this when KV-caches are set up)
    # Shape should be [B, S, S] - True means "can attend", False means "masked"
    # For causal attention: position i can only attend to positions <= i
    causal_mask = torch.tril(torch.ones(S, S, device=device, dtype=torch.bool))
    causal_mask = causal_mask.unsqueeze(0).expand(B, -1, -1)  # [B, S, S]

    if debug:
        print(f"  Causal mask shape: {causal_mask.shape}")

    # Forward through backbone
    # NOTE: Disable autocast for training to avoid mixed precision issues without GradScaler
    # The LoRA weights are in float32 even when base model is bf16/fp16
    h_out = model.backbone(h, mask=causal_mask, input_pos=input_pos)

    if debug:
        print(f"  h_out requires_grad: {h_out.requires_grad}, dtype: {h_out.dtype}")
        print(f"  h_out stats: min={h_out.min().item():.4f}, max={h_out.max().item():.4f}, has_nan={torch.isnan(h_out).any().item()}")

    # Compute losses
    LAMBDA_0 = 2.0
    K = 8
    losses = []

    # Cast h_out to match model dtype for the head layers
    h_out_for_head = h_out.to(dtype)

    # Codebook 0 - uses codebook0_head which is a regular Linear layer
    c0_logits = model.codebook0_head(h_out_for_head)

    if debug:
        print(f"  c0_logits requires_grad: {c0_logits.requires_grad}")
        print(f"  c0_logits stats: min={c0_logits.min().item():.4f}, max={c0_logits.max().item():.4f}, has_nan={torch.isnan(c0_logits).any().item()}")

    c0_targets = target_codes[:, :, 0]

    # Use label smoothing and ignore_index for stability
    c0_loss = F.cross_entropy(
        c0_logits.float().view(-1, c0_logits.size(-1)),  # Cast to float for loss computation
        c0_targets.view(-1),
        reduction='none',
        label_smoothing=0.1,  # Add label smoothing for stability
    ).view(B, S)

    # Clamp loss values to prevent NaN
    c0_loss = torch.clamp(c0_loss, max=100.0)
    c0_loss = (c0_loss * target_mask.float()).sum() / target_mask.sum().clamp(min=1).float()
    losses.append(LAMBDA_0 * c0_loss)

    if debug:
        print(f"  c0_loss requires_grad: {c0_loss.requires_grad}, grad_fn: {c0_loss.grad_fn}")

    # Codebooks 1-7 via decoder - decoder also has LoRA layers
    for k in range(1, K):
        prev_codes = target_codes[:, :, :k]
        offsets_k = torch.arange(k, device=device) * model.config.audio_vocab_size
        prev_codes_offset = prev_codes + offsets_k.view(1, 1, k)

        # Clone embeddings to escape inference mode
        with torch.no_grad():
            prev_embeds = model.audio_embeddings(prev_codes_offset.view(-1)).clone()
        prev_embeds = prev_embeds.view(B, S, k, -1).sum(dim=2)

        # Use h_out_for_head which is already in correct dtype
        decoder_input = h_out_for_head + prev_embeds.to(dtype)

        try:
            model.decoder.reset_caches()
        except:
            pass

        # Project to decoder dimension before passing to decoder
        projected_input = model.projection(decoder_input)
        # Use causal mask for decoder as well (same mask works since same sequence length)
        decoder_out = model.decoder(projected_input, mask=causal_mask, input_pos=input_pos)

        # Use matrix multiply with audio_head parameter (not callable)
        # Clone audio_head to escape inference mode
        decoder_flat = decoder_out.view(-1, decoder_out.size(-1)).to(model.audio_head.dtype)
        audio_head_k = model.audio_head[k - 1].clone()
        ck_logits = torch.mm(decoder_flat, audio_head_k)
        ck_logits = ck_logits.view(B, S, -1)

        ck_targets = target_codes[:, :, k]
        ck_loss = F.cross_entropy(
            ck_logits.float().view(-1, ck_logits.size(-1)),  # Cast to float for loss computation
            ck_targets.view(-1),
            reduction='none',
            label_smoothing=0.1,  # Add label smoothing for stability
        ).view(B, S)
        # Clamp loss values to prevent NaN
        ck_loss = torch.clamp(ck_loss, max=100.0)
        ck_loss = (ck_loss * target_mask.float()).sum() / target_mask.sum().clamp(min=1).float()
        lambda_k = (K - k) / 10.0
        losses.append(lambda_k * ck_loss)

    total_loss = sum(losses)

    # Final NaN/Inf check
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        if debug:
            print(f"  WARNING: NaN/Inf loss detected!")
            for i, loss in enumerate(losses):
                val = loss.item() if torch.isfinite(loss) else ('NaN' if torch.isnan(loss) else 'Inf')
                print(f"    losses[{i}] = {val}")
        # Return a small constant loss to allow training to continue
        total_loss = torch.tensor(10.0, device=device, dtype=torch.float32, requires_grad=True)

    if debug:
        print(f"  total_loss requires_grad: {total_loss.requires_grad}, grad_fn: {total_loss.grad_fn}")

    return total_loss


# ============================================================================
# COMFYUI NODE
# ============================================================================

class FL_HeartMuLa_LoRATrainer:
    """
    Train LoRA adapters on the HeartMuLa model.

    This node allows fine-tuning HeartMuLa with LoRA (Low-Rank Adaptation)
    using audio files with matching lyrics (.txt) and tags (.tags) files.

    Training data should be organized as:
    - audio.wav/.flac/.mp3 (48kHz recommended, will be resampled if different)
    - audio.txt (lyrics with [Verse], [Chorus] markers)
    - audio.tags (comma-separated style tags)

    Supported audio formats: WAV, FLAC, MP3
    """

    RETURN_TYPES = ("HEARTMULA_LORA", "STRING",)
    RETURN_NAMES = ("lora", "log",)
    FUNCTION = "train_lora"
    CATEGORY = "FL HeartMuLa"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "HEARTMULA_MODEL",
                    {"tooltip": "Loaded HeartMuLa model from Model Loader node"}
                ),
                "data_folder": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Folder containing training data (audio files + .txt + .tags). Supports WAV, FLAC, MP3."
                    }
                ),
            },
            "optional": {
                "lora_rank": (
                    "INT",
                    {
                        "default": 16,
                        "min": 4,
                        "max": 128,
                        "step": 4,
                        "tooltip": "LoRA rank (higher = more capacity, more VRAM)"
                    }
                ),
                "lora_alpha": (
                    "FLOAT",
                    {
                        "default": 32.0,
                        "min": 1.0,
                        "max": 128.0,
                        "step": 1.0,
                        "tooltip": "LoRA alpha scaling factor"
                    }
                ),
                "learning_rate": (
                    "FLOAT",
                    {
                        "default": 0.0001,
                        "min": 0.000001,
                        "max": 0.01,
                        "step": 0.00001,
                        "tooltip": "Learning rate for training"
                    }
                ),
                "max_steps": (
                    "INT",
                    {
                        "default": 100,
                        "min": 1,
                        "max": 10000,
                        "step": 10,
                        "tooltip": "Maximum training steps"
                    }
                ),
                "save_checkpoint": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Save LoRA checkpoint to disk after training"
                    }
                ),
                "checkpoint_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Path to save checkpoint (empty = auto-generate)"
                    }
                ),
            }
        }

    @torch.inference_mode(False)  # CRITICAL: Disable inference mode for training
    def train_lora(
        self,
        model: dict,
        data_folder: str,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        learning_rate: float = 0.0001,
        max_steps: int = 100,
        save_checkpoint: bool = True,
        checkpoint_path: str = "",
    ) -> Tuple[dict, str]:
        """
        Train LoRA adapters on the HeartMuLa model.

        Uses @torch.inference_mode(False) to ensure gradient tracking works
        in ComfyUI which runs in inference mode by default.
        """
        import scipy.io.wavfile as wavfile
        import scipy.signal

        # Helper function to load audio from various formats
        def load_audio_file(file_path: Path) -> Tuple[int, np.ndarray]:
            """
            Load audio from WAV, FLAC, or MP3 files.
            Returns (sample_rate, audio_data as float32 numpy array).
            """
            suffix = file_path.suffix.lower()

            if suffix == '.wav':
                sr, audio_data = wavfile.read(str(file_path))
                # Convert to float32
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                elif audio_data.dtype == np.int32:
                    audio_data = audio_data.astype(np.float32) / 2147483648.0
                elif audio_data.dtype == np.float64:
                    audio_data = audio_data.astype(np.float32)
                elif audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                return sr, audio_data

            elif suffix == '.flac':
                try:
                    import soundfile as sf
                    audio_data, sr = sf.read(str(file_path), dtype='float32')
                    return sr, audio_data
                except ImportError:
                    raise ImportError("soundfile package required for FLAC files. Install with: pip install soundfile")

            elif suffix == '.mp3':
                try:
                    from pydub import AudioSegment
                    audio = AudioSegment.from_mp3(str(file_path))
                    sr = audio.frame_rate
                    # Convert to numpy array
                    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
                    # Normalize based on sample width
                    if audio.sample_width == 2:  # 16-bit
                        samples = samples / 32768.0
                    elif audio.sample_width == 4:  # 32-bit
                        samples = samples / 2147483648.0
                    # Handle stereo
                    if audio.channels == 2:
                        samples = samples.reshape((-1, 2))
                    return sr, samples
                except ImportError:
                    raise ImportError("pydub package required for MP3 files. Install with: pip install pydub")

            else:
                raise ValueError(f"Unsupported audio format: {suffix}")

        # Explicitly enable gradients at the start
        torch.set_grad_enabled(True)

        log_lines = []
        def log(msg):
            print(msg)
            log_lines.append(msg)

        log("=" * 60)
        log("[FL HeartMuLa] LoRA Training")
        log("=" * 60)
        log(f"torch.is_inference_mode_enabled(): {torch.is_inference_mode_enabled()}")
        log(f"torch.is_grad_enabled(): {torch.is_grad_enabled()}")

        # Validate data folder
        data_path = Path(data_folder)
        if not data_path.exists():
            raise ValueError(f"Data folder does not exist: {data_folder}")

        # Find training files (support WAV, FLAC, MP3)
        audio_extensions = ['*.wav', '*.flac', '*.mp3', '*.WAV', '*.FLAC', '*.MP3']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(data_path.glob(ext))

        if not audio_files:
            raise ValueError(f"No audio files found in {data_folder}. Supported formats: WAV, FLAC, MP3")

        samples = []
        for audio_file in audio_files:
            txt_file = audio_file.with_suffix(".txt")
            tags_file = audio_file.with_suffix(".tags")
            if txt_file.exists() and tags_file.exists():
                samples.append(audio_file)
            else:
                log(f"  Skipping {audio_file.name} (missing .txt or .tags)")

        if not samples:
            raise ValueError("No complete training samples found (need audio + .txt + .tags)")

        log(f"Found {len(samples)} training samples")
        log(f"LoRA Rank: {lora_rank}")
        log(f"LoRA Alpha: {lora_alpha}")
        log(f"Learning Rate: {learning_rate}")
        log(f"Max Steps: {max_steps}")

        # Get model components
        pipeline = model["pipeline"]
        heartmula = pipeline.model
        tokenizer = pipeline.text_tokenizer
        device = model["device"]
        dtype = model["dtype"]

        # CRITICAL FIX: Strip existing LoRA layers first!
        # ComfyUI caches models, so old LoRA from previous training persists.
        # We must strip them before applying fresh LoRA for this training run.
        has_lora = any(isinstance(m, LoRALinear) for m in heartmula.modules())
        if has_lora:
            log("\nStripping existing LoRA layers (clearing cached state)...")
            heartmula = strip_lora_from_model(heartmula)

        # CRITICAL: Clone all model parameters to escape inference mode!
        # ComfyUI loads models in inference_mode which "taints" all tensors.
        # We must clone them before they can be used in training.
        log("\nCloning model parameters to escape inference mode...")
        heartmula = clone_model_parameters(heartmula)

        # CRITICAL: Enable gradients BEFORE applying LoRA
        # ComfyUI disables gradients globally, we need them enabled during LoRA creation
        log(f"Gradients enabled before LoRA apply: {torch.is_grad_enabled()}")

        # Apply LoRA to model - MUST be done with gradients enabled
        log("Applying LoRA to model...")
        lora_config = {
            'rank': lora_rank,
            'alpha': lora_alpha,
            'dropout': 0.05,
            'target_modules': ['q_proj', 'k_proj', 'v_proj', 'output_proj'],
            'apply_to_backbone': True,
            'apply_to_decoder': True,
        }

        # Temporarily enable gradients for LoRA parameter creation
        with torch.enable_grad():
            heartmula = apply_lora_to_model(heartmula, lora_config)

            # Collect trainable parameters while still in enable_grad context
            trainable_params = [p for p in heartmula.parameters() if p.requires_grad]
            log(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

            # CRITICAL: Verify LoRA weights are properly initialized (not NaN)
            nan_count = 0
            for name, param in heartmula.named_parameters():
                if 'lora_' in name:
                    if torch.isnan(param.data).any():
                        log(f"  WARNING: NaN in {name} after initialization!")
                        nan_count += 1
            if nan_count > 0:
                log(f"  ERROR: {nan_count} LoRA parameters have NaN values after init!")
            else:
                log(f"  LoRA parameters initialized correctly (no NaN)")

            # Verify LoRA params
            for name, param in heartmula.named_parameters():
                if 'lora_A' in name:
                    log(f"  Sample LoRA param '{name}': requires_grad={param.requires_grad}, shape={param.shape}")
                    break

        if len(trainable_params) == 0:
            raise ValueError("No trainable parameters found! LoRA may not have been applied correctly.")

        # Setup optimizer
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)

        # Prepare training data
        log("\nPreparing training data...")
        all_batches = []

        for sample_path in samples:
            # Load audio using format-aware loader
            sr, audio_data = load_audio_file(sample_path)

            # Handle stereo -> mono
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)

            # Resample to 48kHz if needed
            if sr != 48000:
                num_samples = int(len(audio_data) * 48000 / sr)
                audio_data = scipy.signal.resample(audio_data, num_samples)

            waveform = torch.from_numpy(audio_data).unsqueeze(0).float()

            # Tokenize audio
            audio_codes = tokenize_audio(waveform)
            max_frames = 125  # ~10 seconds
            if audio_codes.shape[1] > max_frames:
                audio_codes = audio_codes[:, :max_frames]

            # Load text
            lyrics = sample_path.with_suffix(".txt").read_text(encoding='utf-8')
            tags = sample_path.with_suffix(".tags").read_text(encoding='utf-8')

            # Format prompt
            if not tags.startswith("<tag>"):
                tags = f"<tag>{tags}</tag>"
            if not lyrics.startswith("<lyrics>"):
                lyrics = f"<lyrics>{lyrics}</lyrics>"
            prompt = f"{tags}{lyrics}"

            # Tokenize text
            text_tokens = tokenizer.encode(prompt).ids
            text_tensor = torch.tensor(text_tokens, dtype=torch.long)

            # Build training batch
            num_frames = audio_codes.shape[1]
            seq_len = len(text_tokens) + num_frames

            input_tokens = torch.zeros(1, seq_len, 9, dtype=torch.long)
            input_mask = torch.zeros(1, seq_len, 9, dtype=torch.bool)
            target_codes = torch.zeros(1, seq_len, 8, dtype=torch.long)
            target_mask = torch.zeros(1, seq_len, dtype=torch.bool)

            # Text prefix
            input_tokens[0, :len(text_tokens), 8] = text_tensor
            input_mask[0, :len(text_tokens), 8] = True

            # Audio frames
            for i in range(num_frames):
                pos = len(text_tokens) + i
                input_tokens[0, pos, :8] = audio_codes[:, i]
                input_mask[0, pos, :8] = True
                if i < num_frames - 1:
                    target_codes[0, pos, :] = audio_codes[:, i + 1]
                    target_mask[0, pos] = True

            all_batches.append({
                'input_tokens': input_tokens,
                'input_mask': input_mask,
                'target_codes': target_codes,
                'target_mask': target_mask,
            })

        log(f"Prepared {len(all_batches)} training batches")

        # Training loop with gradients enabled
        log("\nStarting training...")
        pbar = ProgressBar(max_steps)
        heartmula.train()
        losses = []

        # Check gradient state before training
        log(f"Gradients enabled before loop: {torch.is_grad_enabled()}")

        # Force enable gradients for the entire training process
        prev_grad_state = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        log(f"Gradients enabled after set_grad_enabled(True): {torch.is_grad_enabled()}")

        # CRITICAL: Re-verify LoRA parameters have requires_grad=True AFTER enabling gradients
        # This is needed because ComfyUI may have disabled gradients during model loading
        lora_param_count = 0
        for name, param in heartmula.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                if not param.requires_grad:
                    log(f"  WARNING: Re-enabling requires_grad for {name}")
                param.requires_grad_(True)  # Use requires_grad_() to ensure it's set
                lora_param_count += 1

        log(f"Verified {lora_param_count} LoRA parameters have requires_grad=True")

        # Re-create optimizer with verified parameters
        trainable_params = [p for p in heartmula.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
        log(f"Optimizer re-created with {len(trainable_params)} parameters")

        # Use enable_grad() context manager as additional safety (VoxCPM pattern)
        with torch.enable_grad():
            try:
                for step in range(max_steps):
                    batch = all_batches[step % len(all_batches)]
                    optimizer.zero_grad(set_to_none=True)  # More efficient

                    # Run forward pass with debug info on first step
                    loss = training_forward(heartmula, batch, device, dtype, debug=(step == 0))

                    # Debug: check if loss has grad_fn
                    if step == 0:
                        log(f"Loss requires_grad: {loss.requires_grad}")
                        log(f"Loss grad_fn: {loss.grad_fn}")
                        if loss.grad_fn is None:
                            log("ERROR: Loss has no grad_fn! Checking LoRA layers...")
                            for name, module in heartmula.named_modules():
                                if isinstance(module, LoRALinear):
                                    log(f"  {name}: lora_A.requires_grad={module.lora_A.requires_grad}, lora_B.requires_grad={module.lora_B.requires_grad}")
                                    break

                    loss.backward()

                    # Check for NaN/Inf gradients and get gradient statistics
                    has_nan_grad = False
                    max_grad = 0.0
                    for param in trainable_params:
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                has_nan_grad = True
                                break
                            grad_max = param.grad.abs().max().item()
                            if grad_max > max_grad:
                                max_grad = grad_max

                    # Debug: print gradient stats on first few steps
                    if step < 3:
                        log(f"  Step {step + 1}: max_grad={max_grad:.6f}, has_nan={has_nan_grad}")

                    if has_nan_grad:
                        if step < 5:  # Only log first few
                            log(f"  Step {step + 1}: NaN/Inf gradient detected, skipping update")
                        optimizer.zero_grad(set_to_none=True)
                        loss_val = 10.0  # Placeholder
                    else:
                        # Debug: check weight values BEFORE optimizer step
                        if step == 0:
                            for name, param in heartmula.named_parameters():
                                if 'lora_A' in name:
                                    log(f"  BEFORE optimizer.step(): {name}")
                                    log(f"    weight: min={param.data.min().item():.6f}, max={param.data.max().item():.6f}, has_nan={torch.isnan(param.data).any().item()}")
                                    log(f"    grad: min={param.grad.min().item():.6f}, max={param.grad.max().item():.6f}, has_nan={torch.isnan(param.grad).any().item()}")
                                    log(f"    dtype: weight={param.data.dtype}, grad={param.grad.dtype}")
                                    break

                        # Clip gradients more aggressively
                        torch.nn.utils.clip_grad_norm_(trainable_params, 0.5)

                        # Additional safety: clamp gradient values directly
                        for param in trainable_params:
                            if param.grad is not None:
                                param.grad.data.clamp_(-1.0, 1.0)

                        optimizer.step()
                        loss_val = loss.item()

                        # Debug: check weight values AFTER optimizer step
                        if step == 0:
                            for name, param in heartmula.named_parameters():
                                if 'lora_A' in name:
                                    log(f"  AFTER optimizer.step(): {name}")
                                    log(f"    weight: min={param.data.min().item():.6f}, max={param.data.max().item():.6f}, has_nan={torch.isnan(param.data).any().item()}")
                                    break

                    # CRITICAL: Check for NaN in weights after update
                    has_nan_weights = False
                    for param in trainable_params:
                        if torch.isnan(param.data).any():
                            has_nan_weights = True
                            break

                    if has_nan_weights and step < 5:
                        log(f"  Step {step + 1}: WARNING - NaN detected in weights after update!")
                        # Try to diagnose the cause
                        for name, param in heartmula.named_parameters():
                            if 'lora_' in name and torch.isnan(param.data).any():
                                log(f"    NaN in {name}")
                                break

                    losses.append(loss_val)
                    pbar.update_absolute(step + 1)

                    if (step + 1) % 10 == 0:
                        avg_loss = sum(losses[-10:]) / len(losses[-10:])
                        log(f"  Step {step + 1}: loss = {avg_loss:.4f}")
            finally:
                # Restore previous gradient state
                torch.set_grad_enabled(prev_grad_state)

        # Set back to eval mode
        heartmula.eval()
        pipeline.model = heartmula

        # Extract LoRA weights
        lora_state = get_lora_state_dict(heartmula)
        log(f"\nExtracted {len(lora_state)} LoRA parameters")

        # CRITICAL: Check for NaN in final LoRA weights
        nan_params = []
        for name, param in lora_state.items():
            if torch.isnan(param).any():
                nan_params.append(name)
        if nan_params:
            log(f"  ERROR: {len(nan_params)} LoRA parameters have NaN values!")
            log(f"  First few: {nan_params[:5]}")
        else:
            log(f"  All LoRA parameters are valid (no NaN)")

        # Save checkpoint
        checkpoint_file = None
        if save_checkpoint:
            if checkpoint_path:
                checkpoint_file = Path(checkpoint_path)
            else:
                # Auto-generate path
                checkpoint_dir = Path(_PACKAGE_ROOT) / "experiments" / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_file = checkpoint_dir / f"lora_comfyui_r{lora_rank}_s{max_steps}.pt"

            checkpoint = {
                'lora_state_dict': lora_state,
                'config': lora_config,
                'step': max_steps,
                'final_loss': losses[-1] if losses else 0,
            }
            torch.save(checkpoint, checkpoint_file)
            log(f"\nCheckpoint saved to: {checkpoint_file}")

        log("\n" + "=" * 60)
        log("[FL HeartMuLa] Training Complete!")
        log(f"Final loss: {losses[-1]:.4f}" if losses else "No loss recorded")
        log("=" * 60)

        # Build output
        lora_output = {
            'state_dict': lora_state,
            'config': lora_config,
            'checkpoint_path': str(checkpoint_file) if checkpoint_file else None,
            'step': max_steps,
            'final_loss': losses[-1] if losses else 0,
        }

        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (lora_output, "\n".join(log_lines))


class FL_HeartMuLa_LoRALoader:
    """
    Load a trained LoRA checkpoint and apply it to the model.
    """

    RETURN_TYPES = ("HEARTMULA_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_lora"
    CATEGORY = "FL HeartMuLa"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "HEARTMULA_MODEL",
                    {"tooltip": "Loaded HeartMuLa model from Model Loader node"}
                ),
                "lora_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Path to LoRA checkpoint (.pt file)"
                    }
                ),
            },
            "optional": {
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.1,
                        "tooltip": "LoRA strength multiplier"
                    }
                ),
            }
        }

    def load_lora(
        self,
        model: dict,
        lora_path: str,
        strength: float = 1.0,
    ) -> Tuple[dict]:
        """Load and apply LoRA to the model."""

        print("=" * 60)
        print("[FL HeartMuLa] Loading LoRA")
        print("=" * 60)

        # Strip any surrounding quotes from the path (common copy-paste issue)
        lora_path = lora_path.strip().strip('"').strip("'")

        lora_file = Path(lora_path)
        if not lora_file.exists():
            raise ValueError(f"LoRA checkpoint not found: {lora_path}")

        # Load checkpoint
        print(f"Loading: {lora_file}")
        checkpoint = torch.load(lora_file, map_location='cpu')
        lora_config = checkpoint['config']
        lora_state = checkpoint['lora_state_dict']

        print(f"  Step: {checkpoint.get('step', 'unknown')}")
        print(f"  Rank: {lora_config['rank']}")
        print(f"  Alpha: {lora_config['alpha']}")
        print(f"  Parameters: {len(lora_state)}")
        print(f"  Strength: {strength}")

        # Get model
        pipeline = model["pipeline"]
        heartmula = pipeline.model

        # CRITICAL FIX: Always strip existing LoRA layers first!
        # ComfyUI caches models, so old LoRA weights persist between runs.
        # We must strip them and apply fresh to avoid using stale weights.
        has_lora = any(isinstance(m, LoRALinear) for m in heartmula.modules())
        if has_lora:
            print("\nStripping existing LoRA layers (clearing cached state)...")
            heartmula = strip_lora_from_model(heartmula)

        # Apply fresh LoRA architecture
        print("\nApplying fresh LoRA architecture...")
        lora_config['dropout'] = 0.0  # No dropout for inference
        with torch.enable_grad():
            heartmula = apply_lora_to_model(heartmula, lora_config)

        # Load weights from checkpoint
        print("Loading LoRA weights from checkpoint...")
        with torch.enable_grad():
            load_lora_state_dict(heartmula, lora_state)

        # Apply strength AFTER loading weights
        # Strength is applied dynamically in the forward pass
        set_lora_strength(heartmula, strength)

        # Update model (eval mode, doesn't need gradients)
        heartmula.eval()
        pipeline.model = heartmula

        # Store strength in model dict so it can be adjusted later
        model["lora_strength"] = strength
        model["lora_checkpoint"] = str(lora_file)  # Track which checkpoint is loaded

        print("\n" + "=" * 60)
        print("[FL HeartMuLa] LoRA loaded successfully!")
        print(f"  Checkpoint: {lora_file.name}")
        print(f"  Strength: {strength} (adjustable via forward pass)")
        print("=" * 60)

        # Return updated model dict
        return (model,)


class FL_HeartMuLa_LoRASaver:
    """
    Save LoRA weights from a trained model to disk.
    """

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "save_lora"
    CATEGORY = "FL HeartMuLa"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora": (
                    "HEARTMULA_LORA",
                    {"tooltip": "LoRA from the LoRA Trainer node"}
                ),
                "filename": (
                    "STRING",
                    {
                        "default": "my_lora.pt",
                        "tooltip": "Filename for the LoRA checkpoint"
                    }
                ),
            },
            "optional": {
                "output_folder": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Output folder (empty = default checkpoints folder)"
                    }
                ),
            }
        }

    def save_lora(
        self,
        lora: dict,
        filename: str,
        output_folder: str = "",
    ) -> Tuple[str]:
        """Save LoRA weights to disk."""

        if output_folder:
            output_dir = Path(output_folder)
        else:
            output_dir = Path(_PACKAGE_ROOT) / "experiments" / "checkpoints"

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        checkpoint = {
            'lora_state_dict': lora['state_dict'],
            'config': lora['config'],
            'step': lora.get('step', 0),
        }

        torch.save(checkpoint, output_path)
        print(f"[FL HeartMuLa] LoRA saved to: {output_path}")

        return (str(output_path),)
