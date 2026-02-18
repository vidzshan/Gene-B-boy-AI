import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from generator import SasakiGenerator

# ================= CONFIG =================
# We will compare these three versions
CHECKPOINTS = [
    './pretrained_models/generator_checkpoints_hybrid/sasaki_gen_ep40.pt',
    './pretrained_models/generator_checkpoints_hybrid/sasaki_gen_ep50.pt'
]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# NTU Topology (Bones)
BONES = [
    (0, 1), (1, 20), (20, 2), (2, 3),    # Spine
    (20, 4), (4, 5), (5, 6), (6, 7),     # Left Arm
    (20, 8), (8, 9), (9, 10), (10, 11),  # Right Arm
    (0, 12), (12, 13), (13, 14), (14, 15), # Left Leg
    (0, 16), (16, 17), (17, 18), (18, 19)  # Right Leg
]

def load_model(path):
    if not os.path.exists(path):
        print(f"⚠️ Warning: {path} not found. Skipping.")
        return None
    
    model = SasakiGenerator(latent_dim=128, num_classes=3, num_joints=25)
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.to(DEVICE).eval()
    return model

def generate_sample(model):
    # Fixed Noise so we compare "Apples to Apples"
    torch.manual_seed(42) 
    z = torch.randn(1, 128).to(DEVICE)
    
    # Class: Powermove
    lbl = torch.tensor([[0,0,1]]).float().to(DEVICE)
    
    # Intent: HOLD (The hardest test)
    sas = torch.tensor([[0,0,1]]).float().to(DEVICE)
    
    with torch.no_grad():
        out = model(z, lbl, sas) # (1, 3, 64, 25, 1)
        
    motion = out.squeeze().permute(1, 2, 0).cpu().numpy()
    return motion

def compare_visuals(motions, names):
    num_models = len(motions)
    if num_models == 0: return

    fig, axes = plt.subplots(1, num_models, figsize=(5*num_models, 5))
    if num_models == 1: axes = [axes]
    
    fig.suptitle("The Audition: Powermove + Sasaki HOLD (Should be Freeze)", fontsize=16)

    def update(frame):
        for i, ax in enumerate(axes):
            ax.clear()
            ax.set_title(f"{names[i]}\nFrame {frame}")
            ax.set_xlim(-0.8, 0.8)
            ax.set_ylim(-0.8, 0.8)
            ax.axis('off')
            
            motion = motions[i]
            pose = motion[frame]
            x = pose[:, 0]
            y = -pose[:, 1] # Flip Y for visuals
            
            # Bones
            for b in BONES:
                ax.plot([x[b[0]], x[b[1]]], [y[b[0]], y[b[1]]], 'blue', linewidth=2)
            # Joints
            ax.scatter(x, y, c='red', s=30)
            # Head
            ax.scatter(x[3], y[3], c='cyan', s=50)

    ani = animation.FuncAnimation(fig, update, frames=64, interval=50)
    plt.show()

if __name__ == "__main__":
    loaded_motions = []
    loaded_names = []
    
    for path in CHECKPOINTS:
        name = path.split('_')[-1].replace('.pt', '') # e.g., "ep70"
        print(f"Loading {name}...")
        model = load_model(path)
        if model:
            motion = generate_sample(model)
            loaded_motions.append(motion)
            loaded_names.append(name)
            
    compare_visuals(loaded_motions, loaded_names)
