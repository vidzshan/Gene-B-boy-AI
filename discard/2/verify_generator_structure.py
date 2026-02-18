import torch
import numpy as np
from generator import SasakiGenerator

def verify_structure():
    print("--- 1. Checking Generator Architecture ---")
    gen = SasakiGenerator()
    
    # Check 1: Is LayerNorm present?
    if hasattr(gen, 'ln'):
        print("✅ LayerNorm detected. (Prevents Saturation)")
    else:
        print("❌ LayerNorm MISSING. Please update generator.py")
        return

    # Check 2: Initial Output Distribution
    print("\n--- 2. Checking Initial Output (Untrained) ---")
    z = torch.randn(5, 128)
    lbl = torch.zeros(5, 3)
    sas = torch.zeros(5, 3)
    
    with torch.no_grad():
        out = gen(z, lbl, sas) 
    
    # --- FIX: Use reshape() instead of view() ---
    vals = out.reshape(-1).numpy()
    # ----------------------------------------
    
    min_v, max_v = vals.min(), vals.max()
    mean_v = vals.mean()
    
    print(f"Min: {min_v:.4f} (Should be > -0.5)")
    print(f"Max: {max_v:.4f} (Should be < 0.5)")
    print(f"Mean: {mean_v:.4f} (Should be near 0.0)")
    
    # THE BOX CHECK
    count_saturated = ((vals > 0.49) | (vals < -0.49)).sum()
    percent_saturated = count_saturated / len(vals) * 100
    
    print(f"Saturation: {percent_saturated:.2f}% of joints are stuck at the edge.")
    
    if percent_saturated > 10.0:
        print("❌ FAIL: Generator creates a BOX immediately.")
    else:
        print("✅ PASS: Generator produces soft, varied motion.")

if __name__ == "__main__":
    verify_structure()
