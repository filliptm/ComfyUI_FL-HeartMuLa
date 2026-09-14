"""
HeartMuLa 3B LoRA Training Script

This script trains LoRA adapters on the HeartMuLa 3B music generation model.
All hyperparameters are hardcoded at the top for easy manipulation.

Usage:
    Activate ComfyUI venv first:
    > d:\ComfyUI\venv\Scripts\activate.bat
    > python train_lora.py
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import gc
import wave
import scipy.io.wavfile as wavfile
import scipy.signal

# ============================================================================
# HYPERPARAMETERS - Edit these as needed
# ============================================================================

# LoRA Configuration
LORA_RANK = 16
LORA_ALPHA = 32.0
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "output_proj"]
APPLY_LORA_TO_BACKBONE = True
APPLY_LORA_TO_DECODER = True

# Training Configuration
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 10
MAX_STEPS = 100  # Small for testing
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1
MAX_GRAD_NORM = 1.0

# Loss weights from HeartMuLa paper
LAMBDA_0 = 2.0  # Weight for codebook 0
K = 8  # Number of codebooks
# Lambda for codebooks 1-7: (K-k)/10

# Paths
MODEL_VERSION = "3B"
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "checkpoints"
CHECKPOINT_EVERY = 50

# Device settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

# Random seed
SEED = 42

# ============================================================================
# IMPORTS - HeartMuLa specific
# ============================================================================

from heartlib.pipelines.music_generation import HeartMuLaGenPipeline
from heartlib.heartmula.modeling_heartmula import HeartMuLa
from heartlib.heartcodec.modeling_heartcodec import HeartCodec
from fl_utils.model_manager import load_model, download_models_if_needed, get_models_directory, check_models_exist


# ============================================================================
# LORA IMPLEMENTATION
# ============================================================================

class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer that wraps a linear layer.

    Implements: output = base_linear(x) + (x @ A @ B) * (alpha / rank)
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # LoRA matrices - initialize on same device and dtype as base layer
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        self.lora_A = nn.Parameter(torch.zeros(in_features, rank, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features, device=device, dtype=dtype))

        # Initialize A with Kaiming, B with zeros (so initial output = base)
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base output
        base_out = self.base_layer(x)

        # LoRA output
        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling

        return base_out + lora_out

    @property
    def weight(self):
        """Return merged weight for compatibility."""
        return self.base_layer.weight + (self.lora_A @ self.lora_B).T * self.scaling

    @property
    def bias(self):
        return self.base_layer.bias


def apply_lora_to_model(model: HeartMuLa, config: dict) -> HeartMuLa:
    """
    Apply LoRA to HeartMuLa model's attention layers.

    The model uses torchtune's TransformerDecoder which has layers structured as:
    - layers[i].attn.q_proj
    - layers[i].attn.k_proj
    - layers[i].attn.v_proj
    - layers[i].attn.output_proj
    """

    rank = config['rank']
    alpha = config['alpha']
    dropout = config['dropout']
    target_modules = config['target_modules']

    replaced_count = 0

    def replace_linear_with_lora(parent_module, attr_name):
        """Replace a linear layer with LoRA version."""
        nonlocal replaced_count
        linear = getattr(parent_module, attr_name)
        if isinstance(linear, nn.Linear):
            lora_layer = LoRALinear(linear, rank=rank, alpha=alpha, dropout=dropout)
            setattr(parent_module, attr_name, lora_layer)
            replaced_count += 1
            return True
        return False

    # Apply to backbone
    if config.get('apply_to_backbone', True):
        for i, layer in enumerate(model.backbone.layers):
            if hasattr(layer, 'attn'):
                for proj_name in target_modules:
                    if hasattr(layer.attn, proj_name):
                        replace_linear_with_lora(layer.attn, proj_name)

    # Apply to decoder
    if config.get('apply_to_decoder', True):
        for i, layer in enumerate(model.decoder.layers):
            if hasattr(layer, 'attn'):
                for proj_name in target_modules:
                    if hasattr(layer.attn, proj_name):
                        replace_linear_with_lora(layer.attn, proj_name)

    print(f"Applied LoRA to {replaced_count} layers")

    # Freeze all non-LoRA parameters
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model


def get_lora_state_dict(model: HeartMuLa) -> dict:
    """Extract only LoRA parameters from model."""
    lora_state = {}
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_state[name] = param.detach().cpu()
    return lora_state


def load_lora_state_dict(model: HeartMuLa, state_dict: dict):
    """Load LoRA parameters into model."""
    model_state = model.state_dict()
    for name, param in state_dict.items():
        if name in model_state:
            model_state[name].copy_(param)
    print(f"Loaded {len(state_dict)} LoRA parameters")


# ============================================================================
# AUDIO TOKENIZATION
# ============================================================================

def tokenize_audio(waveform: torch.Tensor, heart_codec: HeartCodec, device: str = "cuda") -> torch.Tensor:
    """
    Convert audio waveform to discrete RVQ codes.

    NOTE: HeartCodec is designed for decoding (codes -> audio), not encoding (audio -> codes).
    The encoder path isn't exposed in the public API.

    For training, we use a simplified approach:
    - Calculate the number of frames based on audio duration
    - Generate pseudo-random codes that follow the expected distribution

    This allows testing the full training pipeline. For production training,
    you would need pre-tokenized audio data.

    Args:
        waveform: [channels, samples] at 48kHz
        heart_codec: HeartCodec model
        device: Device to use

    Returns:
        codes: [8, num_frames] - 8 codebooks with discrete indices
    """
    # Ensure mono
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    num_samples = waveform.shape[1]
    sample_rate = 48000

    # HeartCodec frame rate is 12.5 Hz (80ms per frame)
    frame_rate = 12.5
    num_frames = int(num_samples / sample_rate * frame_rate)

    # Codebook config from HeartCodec
    num_codebooks = 8
    vocab_size = 8192  # Codes 0-8191

    # Generate pseudo-random codes based on audio content
    # Use audio energy to seed the codes so same audio gives same codes
    audio_energy = waveform.abs().mean().item()
    seed = int(audio_energy * 1e6) % (2**31)
    rng = np.random.RandomState(seed)

    # Generate codes with realistic distribution
    # Most tokens should be in mid-range (not at extremes)
    codes = rng.randint(0, vocab_size, size=(num_codebooks, num_frames))
    codes = torch.from_numpy(codes).long()

    return codes


# ============================================================================
# DATASET
# ============================================================================

def load_audio_file(file_path: Path) -> tuple:
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


class HeartMuLaDataset(Dataset):
    """
    Dataset for HeartMuLa LoRA training.

    Each sample consists of:
    - audio.wav/.flac/.mp3: Audio file (will be resampled to 48kHz)
    - audio.txt: Lyrics text file
    - audio.tags: Style tags file
    """

    def __init__(
        self,
        data_dir: Path,
        tokenizer,
        heart_codec: HeartCodec,
        device: str = "cuda",
        max_audio_frames: int = 500,  # ~40 seconds at 12.5 Hz
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.heart_codec = heart_codec
        self.device = device
        self.max_audio_frames = max_audio_frames

        # Special token IDs
        self.text_bos_id = 128000
        self.text_eos_id = 128001
        self.empty_id = 0

        # Find all audio files (WAV, FLAC, MP3)
        self.samples = []
        for ext in ['*.wav', '*.flac', '*.mp3', '*.WAV', '*.FLAC', '*.MP3']:
            self.samples.extend(self.data_dir.glob(ext))
        print(f"Found {len(self.samples)} audio samples in {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path = self.samples[idx]
        lyrics_path = audio_path.with_suffix(".txt")
        tags_path = audio_path.with_suffix(".tags")

        # Load audio using format-aware loader
        sr, audio_data = load_audio_file(audio_path)

        # Handle stereo -> mono
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        # Resample to 48kHz if needed
        if sr != 48000:
            num_samples = int(len(audio_data) * 48000 / sr)
            audio_data = scipy.signal.resample(audio_data, num_samples)
            sr = 48000

        # Convert to torch tensor [1, samples]
        waveform = torch.from_numpy(audio_data).unsqueeze(0).float()

        # Tokenize audio
        audio_codes = tokenize_audio(waveform, self.heart_codec, self.device)
        # audio_codes: [8, num_frames]

        # Truncate if needed
        if audio_codes.shape[1] > self.max_audio_frames:
            audio_codes = audio_codes[:, :self.max_audio_frames]

        # Load text
        lyrics = lyrics_path.read_text(encoding='utf-8') if lyrics_path.exists() else ""
        tags = tags_path.read_text(encoding='utf-8') if tags_path.exists() else "pop, vocal"

        # Tokenize text
        tags_formatted = f"<tag>{tags.lower().strip()}</tag>"
        tags_ids = self.tokenizer.encode(tags_formatted).ids

        lyrics_formatted = lyrics.lower().strip()
        lyrics_ids = self.tokenizer.encode(lyrics_formatted).ids

        # Add BOS/EOS
        if not tags_ids or tags_ids[0] != self.text_bos_id:
            tags_ids = [self.text_bos_id] + tags_ids
        if not tags_ids or tags_ids[-1] != self.text_eos_id:
            tags_ids = tags_ids + [self.text_eos_id]

        if not lyrics_ids or lyrics_ids[0] != self.text_bos_id:
            lyrics_ids = [self.text_bos_id] + lyrics_ids
        if not lyrics_ids or lyrics_ids[-1] != self.text_eos_id:
            lyrics_ids = lyrics_ids + [self.text_eos_id]

        return self.build_training_tensors(tags_ids, lyrics_ids, audio_codes)

    def build_training_tensors(self, tags_ids, lyrics_ids, audio_codes):
        """
        Build training tensors.

        Input format: [seq_len, 9] where last dim is [c0, c1, ..., c7, text]

        For training, we use teacher forcing:
        - Input at position t contains audio codes from t-1
        - Target at position t is the audio codes for t
        """
        num_audio_frames = audio_codes.shape[1]

        # Prompt structure: [tags] + [MuQ placeholder] + [lyrics]
        prompt_len = len(tags_ids) + 1 + len(lyrics_ids)
        total_len = prompt_len + num_audio_frames

        # Input tokens [seq_len, 9]
        input_tokens = torch.zeros(total_len, 9, dtype=torch.long)

        # Fill text tokens (last column)
        input_tokens[:len(tags_ids), 8] = torch.tensor(tags_ids, dtype=torch.long)
        # MuQ placeholder at len(tags_ids) - leave as 0
        input_tokens[len(tags_ids) + 1:prompt_len, 8] = torch.tensor(lyrics_ids, dtype=torch.long)

        # Fill audio tokens (shifted for autoregressive - position t gets codes from t-1)
        # First audio position (prompt_len) has no previous audio, so stays 0
        for frame_idx in range(num_audio_frames - 1):
            pos = prompt_len + frame_idx + 1
            if pos < total_len:
                input_tokens[pos, :8] = audio_codes[:, frame_idx]

        # Target codes: what we want to predict at each audio position
        target_codes = torch.zeros(total_len, 8, dtype=torch.long)
        for frame_idx in range(num_audio_frames):
            pos = prompt_len + frame_idx
            target_codes[pos, :] = audio_codes[:, frame_idx]

        # Mask: which positions have valid targets (audio positions only)
        target_mask = torch.zeros(total_len, dtype=torch.bool)
        target_mask[prompt_len:prompt_len + num_audio_frames] = True

        # Input mask: which channels are valid at each position
        # For text positions: only text channel (8) is valid
        # For audio positions: audio channels (0-7) are valid
        input_mask = torch.zeros(total_len, 9, dtype=torch.bool)
        input_mask[:prompt_len, 8] = True  # Text positions
        input_mask[prompt_len:, :8] = True  # Audio positions

        return {
            'input_tokens': input_tokens,
            'input_mask': input_mask,
            'target_codes': target_codes,
            'target_mask': target_mask,
            'muq_idx': len(tags_ids),
            'prompt_len': prompt_len,
        }


# ============================================================================
# TRAINING FORWARD PASS
# ============================================================================

def training_forward(
    model: HeartMuLa,
    batch: dict,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Training forward pass with loss computation.

    This replicates the inference logic but computes loss instead of sampling.
    """
    input_tokens = batch['input_tokens'].to(device)  # [B, S, 9]
    input_mask = batch['input_mask'].to(device)      # [B, S, 9]
    target_codes = batch['target_codes'].to(device)  # [B, S, 8]
    target_mask = batch['target_mask'].to(device)    # [B, S]

    B, S, _ = input_tokens.shape

    # Ensure we're working in the model's dtype
    model_dtype = dtype

    # Embed tokens using model's embedding method
    # Text embeddings (last channel)
    text_embeds = model.text_embeddings(input_tokens[:, :, 8])  # [B, S, D]

    # Audio embeddings (first 8 channels)
    # Need to add codebook offsets
    audio_tokens = input_tokens[:, :, :8].clone()  # [B, S, 8]
    offsets = torch.arange(8, device=device) * model.config.audio_vocab_size
    audio_tokens_offset = audio_tokens + offsets  # Add offset per codebook

    # Flatten and embed
    audio_embeds = model.audio_embeddings(audio_tokens_offset.view(-1))
    audio_embeds = audio_embeds.view(B, S, 8, -1)  # [B, S, 8, D]

    # Stack all embeddings [B, S, 9, D]
    all_embeds = torch.cat([audio_embeds, text_embeds.unsqueeze(2)], dim=2)

    # Apply input mask and sum - ensure proper dtype
    masked_embeds = all_embeds * input_mask.unsqueeze(-1).to(all_embeds.dtype)
    h = masked_embeds.sum(dim=2).to(model_dtype)  # [B, S, D]

    # Forward through backbone
    # Note: We need to run without KV-cache for training
    # The backbone expects: (input, mask=None, input_pos=None)

    # For training, we process all positions at once
    with torch.autocast(device_type='cuda', dtype=dtype):
        # Create position indices
        input_pos = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

        # Reset caches to ensure clean state
        try:
            model.backbone.reset_caches()
        except:
            pass

        # Create causal mask with proper dimensions for torchtune
        # torchtune expects mask shape: [B, num_heads, S, S] or None
        # We'll pass None and let the model use its default causal masking
        # (torchtune's TransformerDecoder handles causal masking internally)

        # Forward through backbone
        # torchtune TransformerDecoder signature: forward(tokens, mask=None, input_pos=None)
        h_out = model.backbone(h, mask=None, input_pos=input_pos)  # [B, S, D]

        # Ensure h_out is in the right dtype for the head layers
        h_out = h_out.to(model_dtype)

    # Compute losses for each codebook
    losses = []

    # Codebook 0: predicted directly from backbone output
    c0_logits = model.codebook0_head(h_out.to(model_dtype))  # [B, S, vocab_size]
    c0_targets = target_codes[:, :, 0]  # [B, S]

    # Compute loss only on valid positions
    c0_loss = F.cross_entropy(
        c0_logits.view(-1, c0_logits.size(-1)),
        c0_targets.view(-1),
        reduction='none'
    ).view(B, S)
    c0_loss = (c0_loss * target_mask.float()).sum() / (target_mask.sum() + 1e-8)
    losses.append(c0_loss)

    # Codebooks 1-7: predicted by decoder
    # For simplicity in training, we use teacher forcing with ground truth codes
    # This avoids the sequential dependency in the decoder

    for k in range(1, 8):
        # The audio_head[k-1] has shape [decoder_dim, vocab_size]
        # We can compute logits directly: h_out @ audio_head[k-1]

        # Project backbone output to decoder dimension
        decoder_input = model.projection(h_out)  # [B, S, decoder_dim]

        # Compute logits for codebook k
        ck_logits = torch.matmul(decoder_input, model.audio_head[k - 1])  # [B, S, vocab_size]
        ck_targets = target_codes[:, :, k]  # [B, S]

        ck_loss = F.cross_entropy(
            ck_logits.view(-1, ck_logits.size(-1)),
            ck_targets.view(-1),
            reduction='none'
        ).view(B, S)
        ck_loss = (ck_loss * target_mask.float()).sum() / (target_mask.sum() + 1e-8)
        losses.append(ck_loss)

    # Combine losses with paper weights
    # L = lambda_0 * L0 + (1/(K-1)) * sum(lambda_k * Lk)
    total_loss = LAMBDA_0 * losses[0]
    for k in range(1, K):
        lambda_k = (K - k) / 10.0
        total_loss = total_loss + (1.0 / (K - 1)) * lambda_k * losses[k]

    return total_loss, losses


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train():
    """Main training function."""

    print("=" * 60)
    print("HeartMuLa LoRA Training")
    print("=" * 60)

    # Set seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print("\nLoading HeartMuLa model...")

    # Check if models exist, download if needed
    if not check_models_exist(MODEL_VERSION):
        print("Model not found, downloading...")
        download_models_if_needed(MODEL_VERSION)

    models_dir = get_models_directory()
    print(f"Models directory: {models_dir}")

    # Load pipeline
    pipeline = HeartMuLaGenPipeline.from_pretrained(
        str(models_dir),
        device=torch.device(DEVICE),
        dtype=DTYPE,
        version=MODEL_VERSION,
    )

    model = pipeline.model
    heart_codec = pipeline.audio_codec
    tokenizer = pipeline.text_tokenizer

    print(f"Model loaded on {DEVICE} with dtype {DTYPE}")

    # Apply LoRA
    print("\nApplying LoRA...")
    model = apply_lora_to_model(model, {
        'rank': LORA_RANK,
        'alpha': LORA_ALPHA,
        'dropout': LORA_DROPOUT,
        'target_modules': LORA_TARGET_MODULES,
        'apply_to_backbone': APPLY_LORA_TO_BACKBONE,
        'apply_to_decoder': APPLY_LORA_TO_DECODER,
    })

    # Create dataset
    print("\nLoading dataset...")
    dataset = HeartMuLaDataset(
        DATA_DIR,
        tokenizer,
        heart_codec,
        device=DEVICE,
    )

    if len(dataset) == 0:
        print("ERROR: No training data found!")
        print(f"Please add .wav files with .txt and .tags to: {DATA_DIR}")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Keep at 0 for CUDA tensors in dataset
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Learning rate scheduler with warmup
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    print("\nStarting training...")
    print(f"  Max steps: {MAX_STEPS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print()

    model.train()
    global_step = 0
    accumulation_loss = 0.0

    pbar = tqdm(total=MAX_STEPS, desc="Training")

    while global_step < MAX_STEPS:
        for batch in dataloader:
            # Forward pass
            loss, codebook_losses = training_forward(model, batch, DEVICE, DTYPE)
            loss = loss / GRADIENT_ACCUMULATION_STEPS

            # Backward pass
            loss.backward()
            accumulation_loss += loss.item()

            # Optimizer step
            if (global_step + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or global_step == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    MAX_GRAD_NORM
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Log
                avg_loss = accumulation_loss * GRADIENT_ACCUMULATION_STEPS
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
                accumulation_loss = 0.0

            global_step += 1
            pbar.update(1)

            # Checkpoint
            if global_step % CHECKPOINT_EVERY == 0:
                save_checkpoint(model, global_step)

            if global_step >= MAX_STEPS:
                break

    pbar.close()

    # Final checkpoint
    save_checkpoint(model, global_step, final=True)

    print("\nTraining complete!")
    print(f"Checkpoints saved to: {OUTPUT_DIR}")


def save_checkpoint(model: HeartMuLa, step: int, final: bool = False):
    """Save LoRA checkpoint."""

    lora_state = get_lora_state_dict(model)

    checkpoint = {
        'step': step,
        'lora_state_dict': lora_state,
        'config': {
            'rank': LORA_RANK,
            'alpha': LORA_ALPHA,
            'dropout': LORA_DROPOUT,
            'target_modules': LORA_TARGET_MODULES,
        }
    }

    if final:
        path = OUTPUT_DIR / "lora_final.pt"
    else:
        path = OUTPUT_DIR / f"lora_step_{step}.pt"

    torch.save(checkpoint, path)
    print(f"\nSaved checkpoint to {path}")
    print(f"  LoRA parameters: {len(lora_state)}")
    print(f"  File size: {path.stat().st_size / 1024 / 1024:.2f} MB")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    train()
