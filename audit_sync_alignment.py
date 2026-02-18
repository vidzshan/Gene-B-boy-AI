
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from audio_brace_loader import AudioBraceDataset

# CONFIG
SKELETON_PATH = './data/brace/BRACE_synced_v2.npz'
AUDIO_DIR = './brace/audio_features/'
AUDIT_SAMPLES = 100

def audit_synchronization():
    print("--- 1. Loading Multi-Modal Dataset for Audit ---")
    dataset = AudioBraceDataset(SKELETON_PATH, AUDIO_DIR)
    
    delays = []
    correlations = []
    
    print(f"--- 2. Auditing {AUDIT_SAMPLES} Random Samples ---")
    indices = np.random.choice(len(dataset), AUDIT_SAMPLES, replace=False)
    
    for idx in tqdm(indices):
        skel, audio, label = dataset[idx]
        
        # Skeleton: (3, 64, 25, 1) -> Kinetic Energy (Velocity)
        # Calculate diff between frames
        skel = skel.squeeze(-1).numpy() # (3, 64, 25)
        # Velocity: magnitude of (x_t+1 - x_t) summed over joints
        diff = np.diff(skel, axis=1) # (3, 63, 25)
        vel = np.linalg.norm(diff, axis=0).sum(axis=-1) # (63,)
        
        # Audio: (64, 33) -> Onset Envelope (Index -1)
        # Matches skel frame rate assumption
        beat = audio[:, -1].numpy()
        beat = beat[:63] # Match velocity length
        
        # Normalize for correlation
        if vel.std() > 1e-6:
            v_n = (vel - vel.mean()) / vel.std()
        else:
            v_n = vel
            
        if beat.std() > 1e-6:
            b_n = (beat - beat.mean()) / beat.std()
        else:
            b_n = beat
        
        # Cross-Correlation
        correlation = np.correlate(v_n, b_n, mode='full')
        lag = np.argmax(correlation) - (len(v_n) - 1)
        
        delays.append(lag)
        correlations.append(correlation)
        
    delays = np.array(delays)
    avg_lag = np.mean(delays)
    std_lag = np.std(delays)
    
    print(f"\n--- AUDIT RESULTS ---")
    print(f"Mean Lag: {avg_lag:.2f} frames")
    print(f"Std Dev Lag: {std_lag:.2f} frames")
    
    # Interpretation
    if abs(avg_lag) < 1.0:
        print("[PASS] Perfect Synchronization (Lag < 1 frame).")
    elif abs(avg_lag) < 3.0:
        print("[WARN] Slight Alignment Drift (1-3 frames). Check FPS matching.")
    else:
        print("[FAIL] Significant Sync Error (> 3 frames). Do not train.")
        
    # Visualize Distribution
    plt.figure(figsize=(10, 5))
    plt.hist(delays, bins=range(-30, 31), color='skyblue', edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', label='Perfect Sync')
    plt.axvline(avg_lag, color='green', linestyle='-', label=f'Mean Lag ({avg_lag:.2f})')
    plt.title("Audio-Motion Synchronization Delay Distribution")
    plt.xlabel("Lag (Frames) - Negative = Motion Early, Positive = Motion Late")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('final_audit_histogram.png')
    print("Saved histogram to final_audit_histogram.png")
    
    with open('lag_result.txt', 'w') as f:
        f.write(f"{avg_lag:.4f}")

if __name__ == "__main__":
    audit_synchronization()
