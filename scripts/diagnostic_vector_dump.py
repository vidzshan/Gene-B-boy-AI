import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from model.generator import AudioAwareGenerator

def audit_vectors(checkpoint_path):
    device = torch.device("cuda")
    gen = AudioAwareGenerator(audio_dim=33).to(device)
    gen.load_state_dict(torch.load(checkpoint_path))
    gen.eval()

    # Generate a random sample
    z = torch.randn(1, 128).to(device)
    audio = torch.randn(1, 64, 33).to(device)
    label = torch.tensor([[1.0, 0.0, 0.0]]).to(device)

    with torch.no_grad():
        output = gen(z, audio, label).cpu().squeeze().numpy() # (3, 64, 25)

    print("--- KINETIC VECTOR AUDIT ---")
    # Check Bone Lengths (Shoulder to Elbow: Joint 20 to 4)
    dist = np.linalg.norm(output[:, :, 20] - output[:, :, 4], axis=0)
    print(f"Mean Bone Length (20->4): {np.mean(dist):.4f}")
    print(f"Bone Length Variance: {np.var(dist):.4f}") # High variance = Jitter/Spaghetti

    # Check Range
    print(f"Coordinate Range: [{np.min(output):.2f}, {np.max(output):.2f}]")
    
    # Dump Frame 0 Vectors for 5 joints
    print("\n--- SAMPLE VECTORS (Frame 0, Joints 0-4) ---")
    print(output[:, 0, :5].T)

if __name__ == "__main__":
    audit_vectors('./pretrained_models/audio_checkpoints/residual_sasaki_gen_e50.pth')