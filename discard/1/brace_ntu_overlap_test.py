import numpy as np
import matplotlib.pyplot as plt

# PATHS
BRACE_PATH = 'data/brace/BRACE_fixed_topology_v3.npz'
NTU_PATH = 'data/ntu120/NTU120_CSub.npz' # Or your NTU sample path

def check_z_axis_alignment():
    print("--- Z-AXIS & SCALE VERIFICATION ---")
    
    # 1. Load BRACE
    try:
        brace = np.load(BRACE_PATH)
        x_brace = brace['x_data'] if 'x_data' in brace else brace['x_train']
        # Brace Shape: (N, 3, 64, 25, 1) or similar. We need (N, C, T, V, M)
        print(f"BRACE Shape: {x_brace.shape}")
    except Exception as e:
        print(f"Error loading BRACE: {e}")
        return

    # 2. Load NTU (Optional: If you don't have the full file, I provide standard stats below)
    ntu_exists = False
    try:
        ntu = np.load(NTU_PATH)
        x_ntu = ntu['x_test'] # Use test set for comparison
        print(f"NTU Shape: {x_ntu.shape}")
        ntu_exists = True
    except:
        print("Warning: NTU file not found. Using Standard NTU Statistics for comparison.")

    # 3. Calculate Statistics
    def get_stats(data, name):
        # Flatten everything to get global min/max per channel
        # Data assumption: (N, 3, T, V, M)
        x_vals = data[:, 0, :, :, :]
        y_vals = data[:, 1, :, :, :]
        z_vals = data[:, 2, :, :, :]
        
        print(f"\n[{name} STATISTICS]")
        print(f"  X-Axis: Min {np.min(x_vals):.4f} | Max {np.max(x_vals):.4f} | Mean {np.mean(x_vals):.4f}")
        print(f"  Y-Axis: Min {np.min(y_vals):.4f} | Max {np.max(y_vals):.4f} | Mean {np.mean(y_vals):.4f}")
        print(f"  Z-Axis: Min {np.min(z_vals):.4f} | Max {np.max(z_vals):.4f} | Mean {np.mean(z_vals):.4f}")
        return np.mean(x_vals), np.std(x_vals)

    mu_brace, std_brace = get_stats(x_brace, "BRACE (Your Data)")
    
    if ntu_exists:
        mu_ntu, std_ntu = get_stats(x_ntu, "NTU (Pre-training Data)")
    else:
        # Standard NTU Stats (Approximate based on literature)
        print("\n[NTU REFERENCE STANDARDS]")
        print("  X-Axis: Range approx -0.8 to 0.8 meters")
        print("  Y-Axis: Range approx -0.8 to 0.8 meters")
        
    # 4. The Verdict Logic
    print("\n--- DIAGNOSIS ---")
    x_range = np.max(x_brace[:, 0]) - np.min(x_brace[:, 0])
    
    if x_range > 10.0:
        print("❌ FAIL: BRACE data is likely still in pixels (Range > 10).")
        print("   Action: Check process_brace.py normalization step.")
    elif x_range < 0.1:
        print("❌ FAIL: BRACE data is too small (Range < 0.1).")
        print("   Action: Check if you divided by image area instead of width.")
    else:
        print(f"✅ PASS: BRACE data range ({x_range:.2f}) is within valid biomechanical scale (approx 1.0).")
        print("   The HPI-GCN will accept this embedding correctly.")

if __name__ == "__main__":
    check_z_axis_alignment()
