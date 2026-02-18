import numpy as np
import torch
from torch.utils.data import Dataset
import os
import glob
import pandas as pd

class AudioBraceDataset(Dataset):
    def __init__(self, skeleton_path, audio_dir, fps=30):
        print(f"Loading Synced Skeleton Data from: {skeleton_path}")
        self.skel_data = np.load(skeleton_path)
        
        self.skeleton_data = self.skel_data['x_data']
        self.video_ids = self.skel_data['video_ids']
        self.seq_indices = self.skel_data['seq_indices']
        self.audio_offsets = self.skel_data['audio_offsets']
        self.labels = self.skel_data['labels']
        
        # Load FPS Metadata (New)
        if 'fps' in self.skel_data:
            self.fps_data = self.skel_data['fps']
        else:
            print("Warning: FPS metadata not found. Defaulting to 30.")
            self.fps_data = np.full(len(self.video_ids), 30.0)

        self.audio_dir = audio_dir
        
        print("Indexing segment audio files...")
        self.audio_map = {}
        all_audio_files = glob.glob(os.path.join(audio_dir, "**", "*.npz"), recursive=True)
        for path in all_audio_files:
            fname = os.path.basename(path)
            parts = fname.replace(".npz", "").split(".")
            if len(parts) >= 2:
                vid_id = parts[0]
                try:
                    seq_idx = int(parts[1])
                    self.audio_map[(vid_id, seq_idx)] = path
                except: continue
        
        print(f"Found {len(self.audio_map)} mapped audio segments.")
        
    def __len__(self):
        return len(self.skeleton_data)

    def __getitem__(self, index):
        skeleton = self.skeleton_data[index]
        label = self.labels[index]
        
        vid_id = self.video_ids[index]
        seq_idx = self.seq_indices[index]
        audio_offset = self.audio_offsets[index]
        video_fps = self.fps_data[index]
        
    def __getitem__(self, index):
        skeleton = self.skeleton_data[index]
        label = self.labels[index]
        
        vid_id = self.video_ids[index]
        seq_idx = self.seq_indices[index]
        audio_offset = self.audio_offsets[index]
        
        audio_path = self.audio_map.get((vid_id, seq_idx))
        TARGET_FRAMES = 64
        
        # Audio Sampling Rate Investigation Result:
        # Majority of dataset features are 1:1 mapped to Video Frames.
        # (25fps video -> 25Hz audio, 30fps video -> 30Hz audio).
        # Therefore, direct indexing is the most robust method.
        # Mean Lag was -1.66 frames. Applying +2 correction.
        
        if audio_path:
            try:
                audio_data = np.load(audio_path)
                
                mfcc = audio_data['mfcc'].squeeze().T 
                chroma = audio_data['chroma_cqt'].squeeze().T
                onset = audio_data['onset_env'].ravel()[:, np.newaxis]
                combined = np.hstack([mfcc, chroma, onset]) # (Total_T, 33)
                
                # Correction based on Audit v5 (Mean Lag -35)
                # Motion is Early -> We need to shift audio retrieval LATER to match.
                SYSTEMIC_CORRECTION = 35 
                
                # Raw start index
                raw_start = audio_offset + SYSTEMIC_CORRECTION
                
                # Safety Clamping (Prevent Crash on Short Audio)
                max_start = len(combined) - TARGET_FRAMES
                start_idx = max(0, min(raw_start, max_start))
                end_idx = start_idx + TARGET_FRAMES
                
                crop = combined[start_idx : end_idx]
                
                # Final check for length (padding if absolutely necessary)
                if len(crop) < TARGET_FRAMES:
                    pad_amt = TARGET_FRAMES - len(crop)
                    crop = np.pad(crop, ((0, pad_amt), (0, 0)), mode='constant')
                
                audio_vector = torch.from_numpy(crop).float()

            except Exception as e:
                audio_vector = torch.zeros(TARGET_FRAMES, 33)
        else:
            audio_vector = torch.zeros(TARGET_FRAMES, 33)

        skeleton = torch.from_numpy(skeleton).float()
        return skeleton, audio_vector, label



if __name__ == "__main__":
    SKELETON_PATH = './data/brace/BRACE_synced_v2.npz'
    AUDIO_DIR = './brace/audio_features/'
    
    if os.path.exists(SKELETON_PATH):
        ds = AudioBraceDataset(SKELETON_PATH, AUDIO_DIR)
        print(f"Dataset Length: {len(ds)}")
        s, a, l = ds[0]
        print(f"Sample 0 Shapes - Skeleton: {s.shape}, Audio: {a.shape}")
        print(f"Sample 0 Meta - Vid: {ds.video_ids[0]}, Seq: {ds.seq_indices[0]}, Offset: {ds.audio_offsets[0]}")
    else:
        print(f"Synced dataset not found. Run build_brace_v2.py first.")


