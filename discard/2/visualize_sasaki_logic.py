import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from generator import SasakiGenerator
from scipy.signal import savgol_filter
from train_brace_final import load_and_split_data
from scipy.spatial.distance import cdist

# CONFIG
MODEL_PATH = "./pretrained_models/generator_checkpoints_hybrid/sasaki_gen_ep50.pt"
LATENT_DIM = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BONES = [ (0, 1), (1, 20), (20, 2), (2, 3), (20, 4), (4, 5), (5, 6), (6, 7), (20, 8), (8, 9), (9, 10), (10, 11), (0, 12), (12, 13), (13, 14), (14, 15), (0, 16), (16, 17), (17, 18), (18, 19) ]

def pipeline(raw_flat, real_db):
    # Motion Match
    dists = cdist(raw_flat, real_db[::5], metric='euclidean')
    matched = real_db[::5][np.argmin(dists, axis=1)].reshape(64, 25, 3)
    # Smooth
    final = np.copy(matched)
    for j in range(25):
        for c in range(3):
            final[:, j, c] = savgol_filter(matched[:, j, c], 9, 2)
    return final

def visualize_logic():
    print("--- DEMONSTRATING SASAKI LOGIC ---")
    gen = SasakiGenerator(latent_dim=LATENT_DIM, num_classes=3, num_joints=25).to(DEVICE)
    gen.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    gen.eval()
    
    # Load DB for cleaning
    X, _, _, _ = load_and_split_data()
    real_db = X.transpose(0, 2, 3, 1, 4).reshape(-1, 75)

    z = torch.randn(1, LATENT_DIM).to(DEVICE)
    label = torch.tensor([[1.0, 0.0, 0.0]]).to(DEVICE) # Toprock

    # --- SCENARIO 1: ATTACK ---
    print("Generating ATTACK Sequence...")
    intent_atk = torch.tensor([[1.0, 0.0, 0.0]]).to(DEVICE)
    with torch.no_grad(): raw_atk = gen(z, label, intent_atk).squeeze().permute(1, 2, 0).cpu().numpy().reshape(64, -1)
    clean_atk = pipeline(raw_atk, real_db)

    # --- SCENARIO 2: HOLD (FREEZE) ---
    print("Generating HOLD Sequence...")
    intent_hold = torch.tensor([[0.0, 0.0, 1.0]]).to(DEVICE)
    with torch.no_grad(): raw_hold = gen(z, label, intent_hold).squeeze().permute(1, 2, 0).cpu().numpy().reshape(64, -1)
    clean_hold = pipeline(raw_hold, real_db)

    # PLOT
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("The SASAKI Test: Does the AI listen to the Brain?", fontsize=16)
    
    ax1.set_title("Intent: ATTACK (High Energy)")
    ax2.set_title("Intent: HOLD (Sasaki Freeze)")
    
    for ax in [ax1, ax2]: ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.8, 0.8); ax.axis('off')

    lines1 = [ax1.plot([], [], 'r-', lw=2)[0] for _ in BONES]
    lines2 = [ax2.plot([], [], 'b-', lw=2)[0] for _ in BONES]

    def update(frame):
        # Attack
        p1 = clean_atk[frame]
        for i, (u, v) in enumerate(BONES): lines1[i].set_data([p1[u,0], p1[v,0]], [p1[u,1], p1[v,1]])
        
        # Hold
        p2 = clean_hold[frame]
        for i, (u, v) in enumerate(BONES): lines2[i].set_data([p2[u,0], p2[v,0]], [p2[u,1], p2[v,1]])
        
        return lines1 + lines2

    ani = animation.FuncAnimation(fig, update, frames=64, interval=75, blit=True)
    plt.show()

if __name__ == "__main__":
    visualize_logic()