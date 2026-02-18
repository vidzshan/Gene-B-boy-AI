import numpy as np
from tqdm import tqdm
import os

# --- CONFIGURATION ---
INPUT_PATH = "data/brace/BRACE_centered_aug.npz"
OUTPUT_PATH = "data/brace/BRACE_fixed_topology_v3.npz"

def fix_skeleton_aggressive(x_data):
    """
    v3: AGGRESSIVE MODE
    1. Reconstructs Spine (Bridge).
    2. FORCEFULLY overwrites fingers/toes to match wrists/ankles.
    """
    N, C, T, V, M = x_data.shape
    fixed_data = x_data.copy()

    print(f"Aggressively processing {N} sequences...")
    
    for n in tqdm(range(N)):
        for t in range(T):
            # Access Person 0
            # pose shape: (3, 25) -> (Channels, Joints)
            
            # --- PART 1: SPINE RECONSTRUCTION (Keep this, it works) ---
            l_hip = fixed_data[n, :, t, 12, 0]
            r_hip = fixed_data[n, :, t, 16, 0]
            l_shldr = fixed_data[n, :, t, 4, 0]
            r_shldr = fixed_data[n, :, t, 8, 0]
            
            # 0: SpineBase
            spine_base = (l_hip + r_hip) / 2
            fixed_data[n, :, t, 0, 0] = spine_base

            # 20: SpineShoulder
            spine_shoulder = (l_shldr + r_shldr) / 2
            fixed_data[n, :, t, 20, 0] = spine_shoulder
            
            # 1: SpineMid
            spine_mid = (spine_base + spine_shoulder) / 2
            fixed_data[n, :, t, 1, 0] = spine_mid
            
            # 2: Neck
            neck_vec = (spine_shoulder - spine_mid)
            neck = spine_shoulder + (neck_vec * 0.4)
            fixed_data[n, :, t, 2, 0] = neck

            # 3: Head (Force Extrapolation just to be safe)
            fixed_data[n, :, t, 3, 0] = neck + (neck_vec * 0.4)

            # --- PART 2: AGGRESSIVE LEAF SNAPPING ---
            # We do NOT check if they are zero. We just overwrite them.
            
            # LEFT ARM (Snap 7, 21, 22 -> Wrist 6)
            wrist_l = fixed_data[n, :, t, 6, 0]
            fixed_data[n, :, t, 7, 0]  = wrist_l  # Hand
            fixed_data[n, :, t, 21, 0] = wrist_l  # HandTip
            fixed_data[n, :, t, 22, 0] = wrist_l  # Thumb

            # RIGHT ARM (Snap 11, 23, 24 -> Wrist 10)
            wrist_r = fixed_data[n, :, t, 10, 0]
            fixed_data[n, :, t, 11, 0] = wrist_r  # Hand
            fixed_data[n, :, t, 23, 0] = wrist_r  # HandTip
            fixed_data[n, :, t, 24, 0] = wrist_r  # Thumb

            # FEET (Snap 15 -> 14, Snap 19 -> 18)
            ankle_l = fixed_data[n, :, t, 14, 0]
            fixed_data[n, :, t, 15, 0] = ankle_l

            ankle_r = fixed_data[n, :, t, 18, 0]
            fixed_data[n, :, t, 19, 0] = ankle_r

    return fixed_data

if __name__ == "__main__":
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Error: {INPUT_PATH} not found.")
        exit()
        
    print(f"Loading {INPUT_PATH}...")
    data = np.load(INPUT_PATH)
    
    # Key handling
    x_key = 'x_data' if 'x_data' in data else 'x_train'
    y_key = 'y_data' if 'y_data' in data else 'y_train'
    
    x_train = data[x_key]
    y_train = data[y_key]
    
    # Run the Aggressive Fix
    x_fixed = fix_skeleton_aggressive(x_train)
    
    # Save
    print(f"Saving Cleaned Topology to {OUTPUT_PATH}...")
    np.savez(OUTPUT_PATH, x_data=x_fixed, y_data=y_train)
    print("✅ Done! All loose joints have been forcefully snapped.")
