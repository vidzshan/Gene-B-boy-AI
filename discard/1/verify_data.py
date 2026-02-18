import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter1d
import os

# ================= CONFIG =================
DATA_PATH = "data/brace/BRACE_centered_aug.npz"
SAVE_PATH = "verify_data4.gif"

# Safe Edges (Same as app.py)
SAFE_EDGES = [
    (4, 8), (12, 16), (4, 12), (8, 16), # Torso Box
    (4, 5), (5, 6), (8, 9), (9, 10),    # Arms
    (12, 13), (13, 14), (16, 17), (17, 18), # Legs
    (3, 4), (3, 8) # Head
]

def visualize_sample(sequence, label, save_path):
    # sequence: [3, 64, 25]
    print(f"Visualizing Sample Label: {label}")
    
    data_x = sequence[0, :, :]
    data_y = sequence[1, :, :]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    def update(frame_idx):
        ax.clear()
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.8, 0.8)
        ax.set_aspect('equal')
        ax.set_title(f"Label: {label} | Frame {frame_idx}")
        ax.grid(True, linestyle='--', alpha=0.3)
        
        x = data_x[frame_idx]
        y = -data_y[frame_idx] # Invert Y
        
        for i, j in SAFE_EDGES:
            ax.plot([x[i], x[j]], [y[i], y[j]], c='blue', linewidth=2)
            
        active = [3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18]
        ax.scatter(x[active], y[active], c='red', s=30)

    ani = animation.FuncAnimation(fig, update, frames=64, interval=50)
    ani.save(save_path, writer='pillow', fps=20)
    print(f"Saved verification GIF to {save_path}")

if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found.")
        exit()
        
    data = np.load(DATA_PATH)
    x = data['x_data'] # (N, 3, 64, 25, 1)
    y = data['y_data']
    
    print(f"Data Shape: {x.shape}")
    print(f"Labels: {np.unique(y)}")
    
    # Pick a random sample
    idx = np.random.randint(0, len(x))
    sample = x[idx].squeeze(-1) # (3, 64, 25)
    label = y[idx]
    
    visualize_sample(sample, label, SAVE_PATH)
