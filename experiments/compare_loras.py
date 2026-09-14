"""Compare LoRA checkpoints to debug training issues."""
import torch

# Load checkpoints
my_lora = torch.load(r"d:\ComfyUI\custom_nodes\ComfyUI_FL-HeartMuLa\experiments\checkpoints\my_lora.pt", map_location="cpu")
good_lora = torch.load(r"d:\ComfyUI\custom_nodes\ComfyUI_FL-HeartMuLa\experiments\checkpoints\lora_final.pt", map_location="cpu")

print("=" * 60)
print("my_lora.pt (from ComfyUI)")
print("=" * 60)
print(f"Step: {my_lora.get('step', 'N/A')}")
print(f"Config: {my_lora.get('config', {})}")
print(f"Num params: {len(my_lora['lora_state_dict'])}")
print("\nFirst 4 parameters:")
for i, (name, param) in enumerate(my_lora['lora_state_dict'].items()):
    if i >= 4:
        break
    print(f"  {name}:")
    print(f"    shape={param.shape}")
    print(f"    mean={param.mean().item():.6f}")
    print(f"    std={param.std().item():.6f}")
    print(f"    max_abs={param.abs().max().item():.6f}")

print("\n" + "=" * 60)
print("lora_final.pt (from standalone script)")
print("=" * 60)
print(f"Step: {good_lora.get('step', 'N/A')}")
print(f"Config: {good_lora.get('config', {})}")
print(f"Num params: {len(good_lora['lora_state_dict'])}")
print("\nFirst 4 parameters:")
for i, (name, param) in enumerate(good_lora['lora_state_dict'].items()):
    if i >= 4:
        break
    print(f"  {name}:")
    print(f"    shape={param.shape}")
    print(f"    mean={param.mean().item():.6f}")
    print(f"    std={param.std().item():.6f}")
    print(f"    max_abs={param.abs().max().item():.6f}")

# Check B matrices specifically (these should be non-zero after training)
print("\n" + "=" * 60)
print("Checking B matrices (should be non-zero after training)")
print("=" * 60)

my_b_zero = 0
my_b_total = 0
for name, param in my_lora['lora_state_dict'].items():
    if 'lora_B' in name:
        my_b_total += 1
        if param.abs().max().item() < 1e-6:
            my_b_zero += 1

good_b_zero = 0
good_b_total = 0
for name, param in good_lora['lora_state_dict'].items():
    if 'lora_B' in name:
        good_b_total += 1
        if param.abs().max().item() < 1e-6:
            good_b_zero += 1

print(f"my_lora.pt: {my_b_zero}/{my_b_total} B matrices are ~zero")
print(f"lora_final.pt: {good_b_zero}/{good_b_total} B matrices are ~zero")
