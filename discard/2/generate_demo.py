import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.spatial.distance import cdist
from scipy.signal import savgol_filter
from generator import SasakiGenerator
from train_brace_final import load_and_split_data
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ================= CONFIG =================
MODEL_PATH = "./pretrained_models/generator_checkpoints_hybrid/sasaki_gen_ep60.pt"
LATENT_DIM = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BONES = [
    (0, 1), (1, 20), (20, 2), (2, 3), (20, 4), (4, 5), (5, 6), (6, 7), 
    (20, 8), (8, 9), (9, 10), (10, 11), (0, 12), (12, 13), (13, 14), 
    (14, 15), (0, 16), (16, 17), (17, 18), (18, 19)
]

TRAIL_JOINTS = [7, 11, 15, 19] 

def load_real_database():
    print(" [1/4] Loading High-Res Motion Library...")
    X_train, _, _, _ = load_and_split_data()
    real_db = X_train.transpose(0, 2, 3, 1, 4).reshape(-1, 25*3)
    return real_db

def advanced_motion_matching(fake_sequence, real_db):
    print(" [2/4] Running High-Fidelity Motion Matching...")
    reconstructed = []
    subset_db = real_db[::2] 
    for frame_idx in range(fake_sequence.shape[0]):
        fake_frame = fake_sequence[frame_idx].reshape(1, -1)
        dists = cdist(fake_frame, subset_db, metric='euclidean')
        best_idx = np.argmin(dists)
        best_frame = subset_db[best_idx]
        reconstructed.append(best_frame)
    return np.array(reconstructed).reshape(64, 25, 3)

def apply_hollywood_smoothing(skeleton):
    print(" [3/4] Applying Savitzky-Golay Kinetic Filter...")
    smoothed = np.copy(skeleton)
    window_length = 9 
    polyorder = 2
    for j in range(25):
        for c in range(3):
            smoothed[:, j, c] = savgol_filter(skeleton[:, j, c], window_length, polyorder)
    return smoothed

def visualize_and_save_gif(raw, fixed, filename="motion_result.gif"):
    print(f" [4/4] Rendering Final Demo to {filename}...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle("Final Architecture Result: Raw Output vs. Decoded Manifold", fontsize=16)

    # --- SETUP AXES ---
    ax1.set_title("Raw Generator Output")
    ax2.set_title("Final Production Output")
    
    for ax in [ax1, ax2]:
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.8, 0.8)
        ax.set_aspect('equal')
        ax.axis('off')

    # --- PLOT OBJECTS ---
    scat1 = ax1.scatter([], [], c='gray', s=10, alpha=0.5)
    lines1 = [ax1.plot([], [], 'gray', alpha=0.5)[0] for _ in BONES]
    
    scat2 = ax2.scatter([], [], c='blue', s=30, zorder=10)
    lines2 = [ax2.plot([], [], 'b-', linewidth=2, zorder=5)[0] for _ in BONES]
    
    trail_len = 8
    history_x = [[] for _ in TRAIL_JOINTS]
    history_y = [[] for _ in TRAIL_JOINTS]
    trail_lines = [ax2.plot([], [], 'c-', linewidth=1, alpha=0.6)[0] for _ in TRAIL_JOINTS]

    def update(frame):
        # Update RAW
        p1 = raw[frame]
        scat1.set_offsets(p1[:, :2])
        for i, (u, v) in enumerate(BONES):
            lines1[i].set_data([p1[u,0], p1[v,0]], [p1[u,1], p1[v,1]])

        # Update FIXED
        p2 = fixed[frame]
        scat2.set_offsets(p2[:, :2])
        for i, (u, v) in enumerate(BONES):
            lines2[i].set_data([p2[u,0], p2[v,0]], [p2[u,1], p2[v,1]])
            if i < 4: lines2[i].set_color('blue')
            elif i < 12: lines2[i].set_color('green')
            else: lines2[i].set_color('red')

        # Update TRAILS
        for idx, joint_idx in enumerate(TRAIL_JOINTS):
            history_x[idx].append(p2[joint_idx, 0])
            history_y[idx].append(p2[joint_idx, 1])
            if len(history_x[idx]) > trail_len:
                history_x[idx].pop(0)
                history_y[idx].pop(0)
            trail_lines[idx].set_data(history_x[idx], history_y[idx])

        return lines1 + lines2 + trail_lines + [scat1, scat2]

    ani = animation.FuncAnimation(fig, update, frames=64, interval=75, blit=True)
    
    # SAVE AS GIF
    # writer='pillow' is the most compatible for GIF saving
    # fps is calculated as 1000 / interval (1000 / 75 ≈ 13)
    ani.save(filename, writer='pillow', fps=13)
    print(f"Successfully saved animation as {filename}")
    plt.close(fig) # Close to free up memory

def main():
    # 1. LOAD MODEL
    gen = SasakiGenerator(latent_dim=LATENT_DIM, num_classes=3, num_joints=25).to(DEVICE)
    try:
        gen.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except:
        print("❌ Model not found. Check path.")
        return
    gen.eval()

    # 2. GENERATE
    print(">>> GENERATING SEQUENCE...")
    z = torch.randn(1, LATENT_DIM).to(DEVICE)
    label = torch.tensor([[0.0, 0.0, 1.0]]).to(DEVICE) 
    intent = torch.tensor([[1.0, 0.0, 0.0]]).to(DEVICE) 
    
    with torch.no_grad():
        raw_output = gen(z, label, intent)
    
    skel_raw = raw_output.squeeze().permute(1, 2, 0).cpu().numpy()
    skel_raw_flat = skel_raw.reshape(64, -1)

    # 3. FIX PIPELINE
    real_db = load_real_database()
    skel_matched = advanced_motion_matching(skel_raw_flat, real_db)
    skel_final = apply_hollywood_smoothing(skel_matched)

    # 4. VISUALIZE AND SAVE
    visualize_and_save_gif(skel_raw, skel_final, "breakdance_output.gif")

if __name__ == "__main__":
    main()