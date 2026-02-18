import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from audio_brace_loader import AudioBraceDataset

# CONFIG
SKELETON_PATH = './data/brace/BRACE_synced_v2.npz'
AUDIO_DIR = './brace/audio_features/'

def normalize(array):
    """Normalize data to 0-1 range for visual comparison"""
    return (array - array.min()) / (array.max() - array.min() + 1e-6)

def analyze_correlation():
    print("--- 1. Loading Multi-Modal Dataset ---")
    dataset = AudioBraceDataset(SKELETON_PATH, AUDIO_DIR)
    
    # Analyze a small batch to find general trends
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    skeletons, audios, labels = next(iter(dataloader))
    
    print("--- 2. Performing Cross-Correlation Audit ---")
    
    # We will plot the first sample to visually verify the 'Soul' of the dance
    plt.figure(figsize=(14, 7))
    
    skel = skeletons[0] # (3, 64, 25, 1)
    audio = audios[0]   # (64, 33)
    
    # A. Calculate Motion Velocity (Energy)
    # Temporal difference between frames
    diff = skel[:, 1:] - skel[:, :-1] 
    # Velocity magnitude across all joints
    velocity = torch.norm(diff, dim=0).sum(dim=(1, 2)) # Shape: (63,)
    
    # B. Extract Music Onset (The Beat)
    # Index 32 is the onset_env based on your diagnostic
    beat = audio[:, -1]
    beat = beat[:63] # Match temporal length
    
    # C. Mathematics: Cross-Correlation
    v_n = normalize(velocity.numpy())
    b_n = normalize(beat.numpy())
    
    # Find the 'Lag' (Is the audio shifted?)
    correlation = np.correlate(v_n, b_n, mode='full')
    lag = np.argmax(correlation) - (len(v_n) - 1)
    
    # D. Visualization
    plt.plot(v_n, label='Kinetic Energy (Dancer Speed)', color='#007bff', linewidth=2)
    plt.plot(b_n, label='Music Onset (Drum Hits)', color='#ff4757', linestyle='--', alpha=0.7)
    
    plt.fill_between(range(len(v_n)), v_n, color='#007bff', alpha=0.1)
    plt.fill_between(range(len(b_n)), b_n, color='#ff4757', alpha=0.1)

    plt.title(f"BRACE Multi-Modal Alignment Audit\nDetected Temporal Lag: {lag} Frames", fontweight='bold')
    plt.xlabel("Time (Frames @ 30fps)")
    plt.ylabel("Normalized Intensity")
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    save_path = "music_motion_alignment.png"
    plt.savefig(save_path, dpi=300)
    print(f"✔ Analysis complete. Saved plot to {save_path}")
    print(f"✔ Optimal Lag: {lag} frames.")
    
    if abs(lag) < 3:
        print("✅ ANALYSIS: The dataset is perfectly synchronized. AI will learn rhythm easily.")
    elif abs(lag) < 8:
        print("⚠️ ANALYSIS: Slight delay detected. AI might look 'lazy' or 'behind the beat'.")
    else:
        print("❌ ANALYSIS: Critical Sync Error. We must apply a shift in the Data Loader.")

if __name__ == "__main__":
    analyze_correlation()