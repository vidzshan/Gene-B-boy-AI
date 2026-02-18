import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import savgol_filter
from model.generator import AudioAwareGenerator

# --- CONFIG ---
DB_PATH = './data/brace/BRACE_synced_v2.npz'
CHECKPOINT = './pretrained_models/audio_checkpoints/residual_sasaki_gen_e50.pth'
MEAN_POSE_PATH = './data/brace/mean_pose.npy'
DEVICE = torch.device("cuda")

class SasakiFinalSystem:
    def __init__(self):
        print("Initializing Neural Retrieval Database...")
        db = np.load(DB_PATH)
        # We extract all real human poses from your database (flattened)
        # Shape: (Total_Frames, 75)
        self.db_poses = db['x_data'].transpose(0, 2, 3, 1, 4).reshape(-1, 75)
        self.db_tensor = torch.from_numpy(self.db_poses).float().to(DEVICE)
        
        print("Loading Sasaki Generator...")
        self.gen = AudioAwareGenerator(audio_dim=33).to(DEVICE)
        self.gen.load_state_dict(torch.load(CHECKPOINT))
        self.gen.eval()
        
        self.mean_pose = torch.from_numpy(np.load(MEAN_POSE_PATH)).float().to(DEVICE).view(1, 3, 1, 25, 1)

    def generate_and_clean(self, audio_features, label_idx):
        with torch.no_grad():
            z = torch.randn(1, 128).to(DEVICE)
            label_oh = torch.zeros(1, 3).to(DEVICE)
            label_oh[0, label_idx] = 1.0
            
            # 1. AI Hallucination (Residual)
            delta = self.gen(z, audio_features, label_oh)
            messy_skel = (self.mean_pose + delta).cpu().squeeze().numpy() # (3, 64, 25)
            
        # 2. Neural Retrieval (RAG)
        # We find the closest 'Real Human' pose for every one of the 64 frames
        query = torch.from_numpy(messy_skel).float().to(DEVICE).permute(1, 2, 0).reshape(64, 75)
        dists = torch.cdist(query, self.db_tensor)
        closest_indices = torch.argmin(dists, dim=1).cpu().numpy()
        
        # Pull real frames
        refined_skel = self.db_poses[closest_indices].reshape(64, 25, 3)
        
        # 3. Savitzky-Golay Smoothing (Temporal Polish)
        # This removes any "snapping" jitter between retrieved poses
        for j in range(25):
            for c in range(3):
                refined_skel[:, j, c] = savgol_filter(refined_skel[:, j, c], 7, 3)
        
        return refined_skel

def run_production():
    system = SasakiFinalSystem()
    
    # Simulate a real audio input (You can load your real .npz features here)
    sample_audio = torch.randn(1, 64, 33).to(DEVICE)
    
    print("Synthesizing Dance...")
    clean_dance = system.generate_and_clean(sample_audio, label_idx=2) # 2 = Powermove
    
    # --- ANIMATION ---
    BONES = [(0,1), (1,20), (20,2), (2,3), (20,4), (4,5), (5,6), (20,8), (8,9), (9,10), (0,12), (12,13), (13,14), (0,16), (16,17), (17,18)]
    fig, ax = plt.subplots(figsize=(6,6))
    
    def update(i):
        ax.clear()
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        # Refined skel is (64, 25, 3) -> [Frame, Joint, Coord]
        x = clean_dance[i, :, 0]
        y = -clean_dance[i, :, 1]
        for u, v in BONES:
            ax.plot([x[u], x[v]], [y[u], y[v]], color='blue', linewidth=2)
        ax.scatter(x, y, color='black', s=10)
        ax.set_title(f"Sasaki Final System | Frame {i}")

    ani = animation.FuncAnimation(fig, update, frames=64, interval=50)
    ani.save('FINAL_DANCE_OUTPUT.gif', writer='pillow')
    print("🚀 SUCCESS: FINAL_DANCE_OUTPUT.gif is ready.")

if __name__ == "__main__":
    run_production()