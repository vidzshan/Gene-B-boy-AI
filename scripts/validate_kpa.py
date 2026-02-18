import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
from model.generator import AudioAwareGenerator

def validate_kpa():
    print("--- KPA STEP 4: Final Residual Inspection (Epoch 50) ---")
    gen = AudioAwareGenerator(audio_dim=33)
    checkpoint_path = './pretrained_models/audio_checkpoints/residual_sasaki_gen_e50.pth'
    
    print(f"Loading Checkpoint: {checkpoint_path}")
    try:
        gen.load_state_dict(torch.load(checkpoint_path))
    except FileNotFoundError:
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        return

    gen.eval()
    
    # Fake Inputs
    z = torch.randn(8, 128)
    aud = torch.randn(8, 64, 33)
    lbl = torch.zeros(8, 3); lbl[:, 0] = 1.0 # Toprock
    
    with torch.no_grad():
        # Output: (B, 3, 64, 25, 1)
        fake_skel = gen(z, aud, lbl)
    
    # Check 1: Coordinate Range
    min_val = fake_skel.min().item()
    max_val = fake_skel.max().item()
    print(f"Coordinate Range: [{min_val:.4f}, {max_val:.4f}]")
    
    if min_val < -2.0 or max_val > 3.0:
        print("❌ FAIL: Coordinates out of Safety Cage [-2.0, 3.0]")
    else:
        print("✅ PASS: Coordinates within Safety Cage.")

    # Check 2: Bone Lengths (Sample: SpineBase->SpineMid (0->1))
    p1 = fake_skel.squeeze(-1) # (B, 3, 64, 25)
    # 0=Base, 1=Mid
    dist = torch.norm(p1[:, :, :, 0] - p1[:, :, :, 1], dim=1)
    mean_bone = dist.mean().item()
    print(f"Mean Bone Length (Spine): {mean_bone:.4f}")
    
    if 0.1 <= mean_bone <= 0.5:
        print("✅ PASS: Bone length plausible.")
    else:
        print(f"❌ FAIL: Bone length implausible (Target 0.1-0.5).")

if __name__ == "__main__":
    validate_kpa()
