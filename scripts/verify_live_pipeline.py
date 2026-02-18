import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import torch
import numpy as np
import cv2
from model.generator import AudioAwareGenerator
from realtime_audio_engine import RealTimeAudioEngine
from sasaki_brain import SasakiLiveBrain

# CONFIG
CHECKPOINT = './pretrained_models/audio_checkpoints/residual_sasaki_gen_e50.pth'
DB_PATH = './data/brace/BRACE_synced_v2.npz'
DEVICE = torch.device("cuda")

def benchmark():
    print("--- 🔍 LIVE SYSTEM COMPONENT BENCHMARK ---")
    
    # 1. SETUP
    t0 = time.time()
    print("1. Loading Models...", end="")
    gen = AudioAwareGenerator(audio_dim=33).to(DEVICE)
    gen.load_state_dict(torch.load(CHECKPOINT))
    gen.eval()
    
    db = np.load(DB_PATH)
    db_poses = db['x_data'].transpose(0, 2, 3, 1, 4).reshape(-1, 75)
    db_tensor = torch.from_numpy(db_poses).float().to(DEVICE)
    
    audio_engine = RealTimeAudioEngine(sr=16000)
    brain = SasakiLiveBrain()
    print(f" DONE ({time.time()-t0:.4f}s)")
    
    # 2. PIPELINE TEST (5 Iterations)
    print("\n2. Running 5 Mock Cycles (Input: Random Noise)")
    
    for i in range(5):
        print(f"\n--- Cycle {i+1} ---")
        
        # A. Audio Processing
        t_start = time.time()
        mock_audio = np.random.uniform(-0.1, 0.1, 8000) # 0.5s at 16kHz
        feat_vec = audio_engine.get_features(mock_audio)
        t_audio = time.time() - t_start
        print(f"  [Audio Extract] : {t_audio*1000:.2f} ms")
        
        # B. Brain
        t_start = time.time()
        style = brain.decide_style(feat_vec)
        t_brain = time.time() - t_start
        print(f"  [Brain Decide]  : {t_brain*1000:.2f} ms")
        
        # C. Generation
        t_start = time.time()
        audio_t = torch.from_numpy(feat_vec).float().to(DEVICE).view(1, 1, 33).repeat(1, 64, 1)
        z = torch.randn(1, 128).to(DEVICE)
        label_oh = torch.zeros(1, 3).to(DEVICE)
        label_oh[0, style] = 1.0
        
        with torch.no_grad():
            messy = gen(z, audio_t, label_oh).squeeze().permute(1,2,0).reshape(64, 75)
        t_gen = time.time() - t_start
        print(f"  [Generator]     : {t_gen*1000:.2f} ms")
        
        # D. Retrieval (Cleanup)
        t_start = time.time()
        dists = torch.cdist(messy[0:1], db_tensor)
        best_idx = torch.argmin(dists)
        clean_pose = db_poses[best_idx.cpu()].reshape(25, 3)
        t_cist = time.time() - t_start
        print(f"  [Neural Search] : {t_cist*1000:.2f} ms")
        
        # E. Total Latency
        total = t_audio + t_brain + t_gen + t_cist
        print(f"  >> TOTAL LATENCY: {total*1000:.2f} ms")
        
        if total > 0.5:
            print("  ❌ CRITICAL: Latency > 0.5s (Will cause buffer overflow)")
        else:
            print("  ✅ OK (Real-time feasible)")

if __name__ == "__main__":
    benchmark()
