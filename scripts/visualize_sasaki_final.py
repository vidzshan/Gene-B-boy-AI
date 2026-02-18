import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from model.generator import AudioAwareGenerator
from audio_brace_loader import AudioBraceDataset

# CONFIG
CHECKPOINT = './pretrained_models/audio_checkpoints/expansion_sasaki_gen_e5.pth' # Update to E14
SKELETON_NPZ = './data/brace/BRACE_synced_v2.npz'
AUDIO_DIR = './brace/audio_features/'
DEVICE = torch.device("cuda")

# NTU BONES
BONES = [(0,1), (1,20), (20,2), (2,3), (20,4), (4,5), (5,6), (20,8), (8,9), (9,10), (0,12), (12,13), (13,14), (0,16), (16,17), (17,18)]

def generate_sample(label_idx, name):
    gen = AudioAwareGenerator(audio_dim=33).to(DEVICE)
    gen.load_state_dict(torch.load(CHECKPOINT))
    gen.eval()

    ds = AudioBraceDataset(SKELETON_NPZ, AUDIO_DIR)
    
    # Find a sample with the matching label to get real audio
    audio = None
    for _ in range(100):
        _, aud, lbl = ds[np.random.randint(len(ds))]
        if lbl == label_idx:
            audio = aud.unsqueeze(0).to(DEVICE)
            break
    
    if audio is None: audio = torch.randn(1, 64, 33).to(DEVICE) # Fallback

    z = torch.randn(1, 128).to(DEVICE)
    label_oh = torch.zeros(1, 3).to(DEVICE)
    label_oh[0, label_idx] = 1.0

    with torch.no_grad():
        fake = gen(z, audio, label_oh).cpu().squeeze().numpy() # (3, 64, 25)

    fig, ax = plt.subplots(figsize=(5,5))
    def update(i):
        ax.clear()
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5) # Wide view for Wide range
        ax.set_aspect('equal')
        x = fake[0, i, :]; y = -fake[1, i, :]
        for u, v in BONES:
            ax.plot([x[u], x[v]], [y[u], y[v]], color='red' if label_idx==2 else 'blue')
        ax.scatter(x, y, color='black', s=5)
        ax.set_title(f"{name} | E14 | Frame {i}")

    ani = animation.FuncAnimation(fig, update, frames=64, interval=50)
    ani.save(f'result_{name}.gif', writer='pillow')
    print(f"✔ Saved: result_{name}.gif")

if __name__ == "__main__":
    generate_sample(0, "Toprock")
    generate_sample(1, "Footwork")
    generate_sample(2, "Powermove")