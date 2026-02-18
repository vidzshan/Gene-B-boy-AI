
import os
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from generator import SasakiGenerator
from audio_brace_loader import AudioBraceDataset

# CONFIG
CHECKPOINT_PATH = './checkpoints/generator_e0.pth' # Checkpoint from epoch 0
OUTPUT_GIF = 'sasaki_gan_result.gif'

BONES = [
    (0, 1), (1, 20), (20, 2), (2, 3), 
    (20, 4), (4, 5), (5, 6), (6, 7), 
    (20, 8), (8, 9), (9, 10), (10, 11), 
    (0, 12), (12, 13), (13, 14), (14, 15), 
    (0, 16), (16, 17), (17, 18), (18, 19)
]

def generate_result():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    gen = SasakiGenerator(num_joints=25).to(device)
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint: {CHECKPOINT_PATH}")
        gen.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    else:
        print(f"WARNING: Checkpoint {CHECKPOINT_PATH} not found. Using random initialized model.")
    
    gen.eval()
    
    # 2. Prepare Inputs
    z = torch.randn(1, 128).to(device)
    # Target class 0 (Toprock) for visualization
    class_label = torch.zeros(1, 3).to(device)
    class_label[0, 0] = 1.0 
    intent = torch.zeros(1, 3).to(device) # Hold intent
    
    # 3. Inference
    with torch.no_grad():
        # (1, 3, 64, 25, 1)
        fake_skel = gen(z, class_label, intent, frames=64)
    
    skel = fake_skel[0, :, :, :, 0].cpu().numpy() # (3, 64, 25)
    return skel

def animate_skel(skel, save_path):
    # skel: (3, 64, 25)
    data_x = skel[0]
    data_y = skel[1]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    def update(frame_idx):
        ax.clear()
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_aspect('equal')
        ax.set_title(f"Sasaki-GAN Generation | Frame {frame_idx}")
        
        x = data_x[frame_idx]
        y = -data_y[frame_idx] # Invert Y
        
        for u, v in BONES:
            ax.plot([x[u], x[v]], [y[u], y[v]], c='blue', linewidth=2)
            
        ax.scatter(x, y, c='black', s=20)
        
    ani = animation.FuncAnimation(fig, update, frames=64, interval=50)
    ani.save(save_path, writer='pillow', fps=20)
    print(f"Saved inference result to {save_path}")

if __name__ == "__main__":
    skel = generate_result()
    animate_skel(skel, OUTPUT_GIF)
