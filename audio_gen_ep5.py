import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from model.generator import AudioAwareGenerator
from audio_brace_loader import AudioBraceDataset
import os

# CONFIG - FIXED FILENAME
CHECKPOINT = './pretrained_models/audio_checkpoints/residual_sasaki_gen_e50.pth'
SKELETON_NPZ = './data/brace/BRACE_synced_v2.npz'
AUDIO_DIR = './brace/audio_features/'
DEVICE = torch.device("cuda")

# NTU BONES
BONES = [(0,1), (1,20), (20,2), (2,3), (20,4), (4,5), (5,6), (20,8), (8,9), (9,10), (0,12), (12,13), (13,14), (0,16), (16,17), (17,18)]

def visualize():
    if not os.path.exists(CHECKPOINT):
        print(f"❌ Error: File not found at {CHECKPOINT}")
        print("Run 'dir .\\pretrained_models\\audio_checkpoints\\' to check the name.")
        return

    gen = AudioAwareGenerator(audio_dim=33).to(DEVICE)
    gen.load_state_dict(torch.load(CHECKPOINT))
    gen.eval()

    ds = AudioBraceDataset(SKELETON_NPZ, AUDIO_DIR)
    # Get a random sample
    _, audio, label = ds[np.random.randint(len(ds))]
    audio = audio.unsqueeze(0).to(DEVICE)
    
    z = torch.randn(1, 128).to(DEVICE)
    label_oh = torch.zeros(1, 3).to(DEVICE)
    label_oh[0, label] = 1.0

    with torch.no_grad():
        # Output shape: (1, 3, 64, 25, 1)
        fake = gen(z, audio, label_oh)
        # Squeeze to (3, 64, 25)
        fake = fake.cpu().squeeze().numpy() 

    # --- POST-PROCESS: SMOOTHING (The "Pro" Look) ---
    try:
        from scipy.signal import savgol_filter
        # Smooth along Time Axis (1). Poly=2 keeps the "snap" of the dance.
        fake = savgol_filter(fake, window_length=9, polyorder=2, axis=1)
        print("✔ Applied Savitzky-Golay Smoothing (Window=9)")
    except ImportError:
        print("⚠ Scipy not found. Skipping smoothing.")
    
    fig, ax = plt.subplots(figsize=(6,6))
    def update(i):
        ax.clear()
        ax.set_xlim(-0.6, 0.6); ax.set_ylim(-0.6, 0.6)
        ax.set_aspect('equal')
        
        # Skeleton data: Channels(3), Frames(64), Joints(25)
        x = fake[0, i, :]
        y = -fake[1, i, :] # Invert Y for display
        
        for u, v in BONES:
            ax.plot([x[u], x[v]], [y[u], y[v]], color='blue', linewidth=2)
        ax.scatter(x, y, color='black', s=15)
        ax.set_title(f"Sasaki-GAN E5 | Sample Label: {label} | Frame {i}")

    ani = animation.FuncAnimation(fig, update, frames=64, interval=50)
    ani.save('check_epoch_5.gif', writer='pillow')
    print("✔ Visual audit saved to check_epoch_5.gif")

if __name__ == "__main__":
    visualize()