import torch
import numpy as np
import sys
from generator import SasakiGenerator

# --- CONFIG ---
MODEL_PATH = "./pretrained_models/generator_checkpoints_ac/sasaki_gen_ep60.pt" 
# ^^^ If ep50 crashed or isn't there, change to ep40.pt
LATENT_DIM = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def analyze_vectors():
    print(f"--- DIAGNOSTIC: ANALYZING {MODEL_PATH} ---")
    
    # 1. Load Model
    try:
        gen = SasakiGenerator(latent_dim=LATENT_DIM, num_classes=3, num_joints=25)
        gen.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        gen.to(DEVICE).eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Generate One Sample (Toprock / Attack)
    z = torch.randn(1, LATENT_DIM).to(DEVICE)
    # Class: Toprock (1, 0, 0)
    label = torch.tensor([[1.0, 0.0, 0.0]]).to(DEVICE)
    # Intent: Attack (1, 0, 0) - High Energy
    intent = torch.tensor([[1.0, 0.0, 0.0]]).to(DEVICE)
    
    with torch.no_grad():
        # Output shape: (1, 3, 64, 25, 1)
        skeleton = gen(z, label, intent)
    
    # 3. Convert to Numpy for printing
    # Shape becomes: (64 frames, 25 joints, 3 coords)
    skel_np = skeleton.squeeze().permute(1, 2, 0).cpu().numpy()
    
    # 4. PRINT DIAGNOSTICS
    
    print("\n=== GLOBAL STATISTICS ===")
    print(f"Min Value: {np.min(skel_np):.4f} (Should be > -0.5)")
    print(f"Max Value: {np.max(skel_np):.4f} (Should be < 0.5)")
    print(f"Mean Value: {np.mean(skel_np):.4f} (Should be near 0.0)")
    
    print("\n=== FRAME 0 (Start Pose) - JOINTS 0 to 5 ===")
    # Format: [X, Y, Z]
    print(skel_np[0, 0:6, :])

    print("\n=== FRAME 30 (Mid Move) - JOINTS 0 to 5 ===")
    print(skel_np[30, 0:6, :])
    
    print("\n=== BONE LENGTH CHECK (Arm: Joint 20->4) ===")
    # 20 is SpineShoulder, 4 is Left Shoulder
    joint_20 = skel_np[:, 20, :]
    joint_4 = skel_np[:, 4, :]
    dist = np.linalg.norm(joint_20 - joint_4, axis=1)
    print(f"Avg Length: {np.mean(dist):.4f}")
    print(f"Max Length: {np.max(dist):.4f}")
    print(f"Min Length: {np.min(dist):.4f}")
    
    print("\n=== COPY PASTE THE BELOW ARRAY FOR ME ===")
    print("RAW_VECTOR_SAMPLE = [")
    # Print just the first 5 frames of Joint 0 (SpineBase) and Joint 3 (Head)
    # to keep the message size small but useful.
    for i in range(5):
        head = skel_np[i, 3, :]
        spine = skel_np[i, 0, :]
        print(f"  [Frame {i}, Spine: {spine.tolist()}, Head: {head.tolist()}],")
    print("]")

if __name__ == "__main__":
    analyze_vectors()