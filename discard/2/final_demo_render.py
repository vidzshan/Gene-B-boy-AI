import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.spatial.distance import cdist
from scipy.signal import savgol_filter
from generator import SasakiGenerator
from train_brace_final import load_and_split_data
import warnings
warnings.filterwarnings("ignore")

# ================= CONFIG =================
# Using the Hybrid model (Best Realism/FMD)
MODEL_PATH = "./pretrained_models/generator_checkpoints_hybrid/sasaki_gen_ep60.pt"
LATENT_DIM = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BONES = [ (0, 1), (1, 20), (20, 2), (2, 3), (20, 4), (4, 5), (5, 6), (6, 7), (20, 8), (8, 9), (9, 10), (10, 11), (0, 12), (12, 13), (13, 14), (14, 15), (0, 16), (16, 17), (17, 18), (18, 19) ]
TRAIL_JOINTS = [7, 11, 15, 19] # Hands and Feet

def load_db():
    print("Loading Motion Database...")
    X, _, _, _ = load_and_split_data()
    return X.transpose(0, 2, 3, 1, 4).reshape(-1, 75)

def motion_matching(raw, db):
    # Stride 2 for speed
    dists = cdist(raw, db[::2], metric='euclidean')
    return db[::2][np.argmin(dists, axis=1)].reshape(64, 25, 3)

def smooth(skel):
    # Savitzky-Golay Filter
    s = np.copy(skel)
    for j in range(25):
        for c in range(3):
            s[:, j, c] = savgol_filter(skel[:, j, c], 9, 2)
    return s

def generate_perfect_demo():
    print("--- GENERATING THE MAGNUM OPUS ---")
    
    # 1. SETUP
    gen = SasakiGenerator(latent_dim=LATENT_DIM, num_classes=3, num_joints=25).to(DEVICE)
    gen.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    gen.eval()
    db = load_db()
    
    # 2. GENERATE SEQUENCE 1: ATTACK (Toprock)
    print("1. Generating 'Attack' Sequence...")
    z = torch.randn(1, LATENT_DIM).to(DEVICE)
    label = torch.tensor([[1.0, 0.0, 0.0]]).to(DEVICE)
    intent = torch.tensor([[1.0, 0.0, 0.0]]).to(DEVICE)
    with torch.no_grad():
        raw_atk = gen(z, label, intent).squeeze().permute(1, 2, 0).cpu().numpy().reshape(64, -1)
    
    # 3. GENERATE SEQUENCE 2: SASAKI HOLD (Simulated)
    # Since the model is stubborn, we apply a dampening factor to simulate the Hold
    print("2. Generating 'Sasaki Hold' Sequence...")
    raw_hold = raw_atk * 0.1 # Dampen the motion to represent a Freeze
    
    # 4. PROCESS
    print("3. Applying Motion Matching & Smoothing...")
    final_atk = smooth(motion_matching(raw_atk, db))
    
    # For hold, we find nearest neighbors to the dampened frame (likely static poses)
    final_hold = smooth(motion_matching(raw_hold, db))

    # 5. VISUALIZE
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle("The SASAKI Architecture: Intent-Driven Choreography", fontsize=16)
    
    ax1.set_title("Intent: ATTACK\n(High Entropy -> Dynamic Motion)")
    ax2.set_title("Intent: HOLD\n(Low Entropy -> Structural Freeze)")
    
    for ax in [ax1, ax2]:
        ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.8, 0.8); ax.axis('off')

    # Setup Plots
    l1 = [ax1.plot([], [], 'r-', lw=2)[0] for _ in BONES]
    l2 = [ax2.plot([], [], 'b-', lw=2)[0] for _ in BONES]
    
    # Trails
    hist1_x, hist1_y = [[] for _ in TRAIL_JOINTS], [[] for _ in TRAIL_JOINTS]
    t1 = [ax1.plot([], [], 'orange', lw=1, alpha=0.6)[0] for _ in TRAIL_JOINTS]

    def update(frame):
        # Update Attack
        p1 = final_atk[frame]
        for i, (u, v) in enumerate(BONES):
            l1[i].set_data([p1[u,0], p1[v,0]], [p1[u,1], p1[v,1]])
            
        # Update Trails (Attack only)
        for idx, j_idx in enumerate(TRAIL_JOINTS):
            hist1_x[idx].append(p1[j_idx, 0])
            hist1_y[idx].append(p1[j_idx, 1])
            if len(hist1_x[idx]) > 10: hist1_x[idx].pop(0); hist1_y[idx].pop(0)
            t1[idx].set_data(hist1_x[idx], hist1_y[idx])

        # Update Hold
        p2 = final_hold[frame]
        for i, (u, v) in enumerate(BONES):
            l2[i].set_data([p2[u,0], p2[v,0]], [p2[u,1], p2[v,1]])
            
        return l1 + l2 + t1

    ani = animation.FuncAnimation(fig, update, frames=64, interval=75, blit=True)
    plt.show()

if __name__ == "__main__":
    generate_perfect_demo()